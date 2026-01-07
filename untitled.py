class CrossAttentionEncoderImuMask(nn.Module):  # Quantiles as Images
    def __init__(self, input_channel, args):
        super(CrossAttentionEncoderImuMask, self).__init__()
        self.context_feature_dim = args.context_feature_dim
        self.kernel_size = args.kernel_size
        self.map_size = args.map_size
        self.imu_hidden_dim = args.imu_hidden_dim
        self.batch_size = args.batch_size
        self.embed_dim = 128 # self.context_feature_dim * int(self.map_size/4) * int(self.map_size/4)

        # Transform IMU features into Query
        self.imu_mlp = nn.Linear(self.imu_hidden_dim, self.embed_dim)
        self.map_mlp = nn.Linear(1, self.embed_dim)

        # CNN for spatial feature extraction (Key, Value)
        self.context_cnn = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=self.kernel_size, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=self.kernel_size, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, self.context_feature_dim, kernel_size=self.kernel_size, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten())

        # Cross-Attention
        self.cross_attention = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=4, batch_first=True)

    def forward(self, imu_lstm_features, feasible_mask):
        # Extract spatial features (Key, Value)
        spatial_features = self.context_cnn(feasible_mask)  # Shape: [batch_size, flattened_dim]
        spatial_features = spatial_features.unsqueeze(-1)  # Shape: [1, batch_size, feature_dim]
        att_mask = (feasible_mask.squeeze() < 0.2)
        att_mask = att_mask.view(self.batch_size, -1)
        # Transform IMU features into Query
        query = self.imu_mlp(imu_lstm_features.mean(dim=1)).unsqueeze(1)  # Shape: [1, batch_size, feature_dim]
        spatial_embed = self.map_mlp(spatial_features)
        #spatial_embed = spatial_embed.permute(1, 0, 2)
        # Apply cross-attention
        attn_output, _ = self.cross_attention(query, spatial_embed, spatial_embed, key_padding_mask=att_mask)
        attn_output = attn_output.squeeze(1)

        return attn_output

class TrajectoryGeneratorDecoder_2(nn.Module):
    def __init__(self, args):
        """
        :param input_dim: Dimensionality of the input (e.g., 2 for 2D positions).
        :param hidden_dim: Hidden size of the LSTM layer.
        :param latent_dim: Size of the latent vector for stochasticity.
        :param output_dim: Output size (2 for x, y coordinates).
        :param num_layers: Number of LSTM layers.
        """
        super(TrajectoryGeneratorDecoder_2, self).__init__()
        self.decoder_hidden_dim = args.decoder_hidden_dim
        self.output_dim = args.output_dim
        self.decoder_layers = args.decoder_layers
        self.embedding_dim = args.embedding_dim
        #self.h_dim = args.imu_hidden_dim # (simple, convlstm, attention) encoder
        self.h_dim = 6 #((CrossAttentionImuMask) Encoder)
        # LSTM cell for step-by-step prediction
        # IMU features as input
        self.embedding = nn.Linear(self.h_dim, self.embedding_dim)

        # encoder attention as input
        self.lstm = nn.LSTM(self.embedding_dim, self.decoder_hidden_dim, self.decoder_layers, batch_first=True)
        # Output layer to generate 2D coordinates
        self.output_layer = nn.Linear(self.decoder_hidden_dim, self.output_dim)
        self.relu = torch.nn.ReLU()
    
    def forward(self, feature_vec, state_tuple):
        """
        :param initial_position: Initial position of shape [N, 2] (x, y coordinates).
        :param encoder_hidden: Hidden state features from the ConvLSTM encoder [N, hidden_dim].
        :param latent_vector: Latent vector for initialization [N, latent_dim].
        :param seq_length: The length of the sequence to generate (number of time steps).
        :return: Generated trajectory of shape [N, seq_length, 2].
        """
        # Step 2: Prepare input position and container for generated trajectory
        decoder_input = self.embedding(feature_vec)
        decoder_input = self.relu(decoder_input)
        output, state_tuple = self.lstm(decoder_input, state_tuple)
        relative_position = self.output_layer(output)
        return relative_position


class Generator(nn.Module):
    def __init__(self, input_channel, dts, args):
        super(Generator, self).__init__()
        self.imu_hidden_dim = args.imu_hidden_dim
        self.map_size = args.map_size
        self.latent_dim = args.latent_dim
        self.decoder_hidden_dim = args.decoder_hidden_dim
        self.batch_size = args.batch_size
        self.mlp_dim = args.mlp_dim
        self.context_feature_dim = args.context_feature_dim
        self.dts = dts
        self.target_type = args.target_type
        self.embed_dim = 128

        # Encoder: ConvLSTM for spatio-temporal feature extraction
        #self.encoder = TrajectoryGeneratorEncoder(input_channel, args)
        #self.encoder  = SimpleEncoder(input_channel, args)
        self.encoder = CrossAttentionEncoderImuMask(input_channel, args)
        #self.encoder = AttentionEncoder(input_channel, args)

        #mlp_decoder_context_dims = [self.context_feature_dim*int(self.map_size/4)*int(self.map_size/4), self.mlp_dim, self.decoder_hidden_dim - self.latent_dim]  # (CrossAttentionImuMask) encoder
        #mlp_decoder_context_dims = [self.decoder_hidden_dim, self.mlp_dim, self.decoder_hidden_dim - self.latent_dim] # (simple, attention) encoder
        #mlp_decoder_context_dims = [self.imu_hidden_dim*int(self.map_size/4)*int(self.map_size/4), self.mlp_dim, self.decoder_hidden_dim - self.latent_dim]  # (convlstm) encoder

        mlp_decoder_context_dims = [self.embed_dim, self.mlp_dim, self.decoder_hidden_dim - self.latent_dim]  # (CrossAttentionImuMask) encoder

        self.mlp_decoder_context = make_mlp(mlp_decoder_context_dims)


        # Decoder full trajectory prediction
        self.decoder = TrajectoryGeneratorDecoder_2(args)

    def get_masks(self, X_quantiles, Y_quantiles, map_mask, initial_position, bounds):
        X_quantiles = relative_to_abs(X_quantiles, initial_position[:, [0]], self.dts, self.target_type)
        Y_quantiles = relative_to_abs(Y_quantiles, initial_position[:, [1]], self.dts, self.target_type)
        lower_x, lower_y = X_quantiles[:, :, 0], Y_quantiles[:, :, 0]
        upper_x, upper_y = X_quantiles[:, :, -1], Y_quantiles[:, :, -1]
        x_min, x_max, y_min, y_max = bounds

        # Method 1 without interpolation
        # Create a grid of all possible indices in the x and y dimensions
        x_full_range = torch.linspace(x_min, x_max, self.map_size, device=DEVICE)
        y_full_range = torch.linspace(y_min, y_max, self.map_size, device=DEVICE)
        x_full_range, y_full_range = torch.meshgrid(x_full_range, y_full_range, indexing='ij')

        # Reshape x_full_range and y_full_range for comparison across batches and sequence length
        x_full_range = x_full_range.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, height, width)
        y_full_range = y_full_range.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, height, width)

        # Expand lower and upper quantiles to match the grid shape for comparison
        lower_x = lower_x.unsqueeze(-1).unsqueeze(-1)  # Shape: (batch_size, L, 1, 1)
        upper_x = upper_x.unsqueeze(-1).unsqueeze(-1)  # Shape: (batch_size, L, 1, 1)
        lower_y = lower_y.unsqueeze(-1).unsqueeze(-1)  # Shape: (batch_size, L, 1, 1)
        upper_y = upper_y.unsqueeze(-1).unsqueeze(-1)  # Shape: (batch_size, L, 1, 1)

        # Use sigmoid for soft boundaries
        temperature = 2  # Adjust to control the smoothness of the boundary
        x_mask = torch.sigmoid(temperature * (x_full_range - lower_x)) * torch.sigmoid(temperature * (upper_x - x_full_range))
        y_mask = torch.sigmoid(temperature * (y_full_range - lower_y)) * torch.sigmoid(temperature * (upper_y - y_full_range))
        mask = x_mask * y_mask
        # Use Gaussian-like soft boundaries
        # sigma = 1.0  # Adjust for smoothness of boundaries
        # x_mask = torch.exp(-((x_full_range - lower_x) ** 2) / (2 * sigma ** 2)) * \
        #          torch.exp(-((x_full_range - upper_x) ** 2) / (2 * sigma ** 2))
        # y_mask = torch.exp(-((y_full_range - lower_y) ** 2) / (2 * sigma ** 2)) * \
        #          torch.exp(-((y_full_range - upper_y) ** 2) / (2 * sigma ** 2))
    
        # # Combine masks
        # mask = x_mask * y_mask  # Combine x and y masks
        # mask = mask.sum(dim=1)  # Smooth summation across sequence length
    
        # # Normalize mask to the range [0, 1]
        # mask = mask / mask.max(dim=1)[0].max(dim=1)[0].unsqueeze(1).unsqueeze(2)  # Ensure values are scaled between 0 and 1
        mask = mask.sum(dim=1)  # Smooth summation across sequence length
        mask = mask / mask.max(dim=1)[0].max(dim=1)[0].unsqueeze(1).unsqueeze(2)
        """
        # For validation
        # Compute precise floating-point grid coordinates
        x_grid_lower = ((lower_x - x_min) / (x_max - x_min)) * (M)
        y_grid_lower = ((lower_y - y_min) / (y_max - y_min)) * (M)

        x_grid_upper = ((upper_x - x_min) / (x_max - x_min)) * (M)
        y_grid_upper = ((upper_y - y_min) / (y_max - y_min)) * (M)
        # Compute nearest integer grid indices for array access
        x_index_lower = x_grid_lower.int().long()
        y_index_lower = y_grid_lower.int().long()

        x_index_upper = x_grid_upper.int().long()
        y_index_upper = y_grid_upper.int().long()

        x_index_lower = torch.clamp(x_index_lower, 0, M - 1)
        y_index_lower = torch.clamp(y_index_lower, 0, M - 1)

        x_index_upper = torch.clamp(x_index_upper, 0, M - 1)
        y_index_upper = torch.clamp(y_index_upper, 0, M - 1)
        """
        mask = mask.unsqueeze(1)  # Shape: [batch_size, 1, height, width]
        #final_mask = (1 - map_mask) * mask
        final_mask = map_mask * mask
        final_mask = final_mask / final_mask.max(dim=2)[0].max(dim=2)[0].unsqueeze(2).unsqueeze(3)
        return final_mask

    def add_noise(self, _input):
        npeds = _input.size(0)
        seq_len = _input.size(1)
        noise_shape = (self.latent_dim,)
        z_decoder = get_noise(noise_shape)
        vec = z_decoder.view(1, self.latent_dim).repeat(npeds, 1)
        return torch.cat((_input, vec), dim=-1)

    def forward(self, map_mask, initial_position, Q_model, imu_data, bounds):
        # Pass IMU data through the quantile model to get upper and lower quantiles
        X_quantiles, Y_quantiles, feature_vector = Q_model(imu_data)
        # Get the plausible region
        #static_mask = self.get_masks(X_quantiles, Y_quantiles, map_mask, initial_position, bounds)

        # Simple encoder, convlstm encoder
        #last_hidden_state = self.encoder(feature_vector, static_mask) # (Simple, convlstm, attention) encoder
        last_hidden_state = self.encoder(feature_vector, map_mask) #(CrossAttentionImuMask Encoder

        #convlstm
        #last_hidden_state = last_hidden_state.view(last_hidden_state.size(0), -1)  # Flatten to [batch_size, hidden_dim * m * m]

        noise_input = self.mlp_decoder_context(last_hidden_state)

        decoder_h = self.add_noise(noise_input)
        decoder_h = torch.unsqueeze(decoder_h, 0)
        decoder_c = torch.zeros((1, self.batch_size, self.decoder_hidden_dim), device=DEVICE)
        state_tuple = (decoder_h, decoder_c)

        #IMU features as input
        #predicted_positions = self.decoder(feature_vector, state_tuple) # (simple, convlstm, attention) encoder
        predicted_positions = self.decoder(torch.cat((X_quantiles, Y_quantiles), dim=-1), state_tuple) #(CrossAttentionImuMask) Encoder

        return predicted_positions