def trainQmodel(args, train_loader, val_loader, log_file, Qtraining_epochs):
    train_mini_batches = len(train_loader)
    val_mini_batches = len(val_loader)

    # Initialize QLSTM model
    if args.model_type == 'Q_LSTM':
        lstm_model = QLSTMModel(input_dim=args.input_dim, output_dim=args.output_dim, args=args).to(args.device)
    else:
        lstm_model = LSTMModel(input_dim=args.input_dim, output_dim=args.output_dim, args=args).to(args.device)


    criterion = loss_fns['LSTM_quantile_loss']
    q_optimizer = torch.optim.Adam(lstm_model.parameters(), args.lr)
    q_scheduler = ReduceLROnPlateau(q_optimizer, 'min', patience=15, factor=0.75, verbose=True, eps=1e-12)

    start_epoch = 0
    best_val_loss = np.inf
    try:
        for epoch in range(start_epoch, Qtraining_epochs):
            log_line = ''
            lstm_model.train()
            train_loss = 0

            for bid, batch in enumerate(train_loader):
                feat, targ, _, _, _ = batch
                feat, targ = feat.to(args.device), targ.to(args.device)
                q_optimizer.zero_grad()
                if args.model_type == 'Q_LSTM':
                    predicted_1, predicted_2, _ = lstm_model(feat)
                    total_loss = criterion(predicted_1, predicted_2, targ)
                    train_loss += total_loss.cpu().detach().numpy()
                else:
                    predicted, _ = lstm_model(feat)
                    total_loss = criterion(predicted, targ)
                    train_loss += total_loss.cpu().detach().numpy()

                total_loss.backward()
                q_optimizer.step()

            train_loss = train_loss / train_mini_batches
            log_line = format_string(log_line, epoch, q_optimizer.param_groups[0]['lr'], train_loss)
            saved_model = False

            if val_loader:
                lstm_model.eval()
                val_loss = 0
                for bid, batch in enumerate(val_loader):
                    feat, targ, _, _, _ = batch
                    feat, targ = feat.to(args.device), targ.to(args.device)
                    q_optimizer.zero_grad()
                    if args.model_type == 'Q_LSTM':
                        pred_1, pred_2, _ = lstm_model(feat)
                        v_loss = criterion(pred_1, pred_2, targ)
                        val_loss += v_loss.cpu().detach().numpy()
                    else:
                        pred, _ = lstm_model(feat)
                        v_loss = criterion(pred, targ)
                        val_loss += v_loss.cpu().detach().numpy()

                val_loss = val_loss / val_mini_batches
                log_line = format_string(log_line, val_loss)

                print(f'Epoch [{epoch}/{Qtraining_epochs}] | Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}')
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    saved_model = True
                    if args.output_directory:
                        model_path = osp.join(args.output_directory, 'checkpoints', 'lstm_checkpoint_%d.pt' % epoch)
                        torch.save({'model_state_dict': lstm_model.state_dict(),
                                    'epoch': epoch,
                                    'loss': train_loss,
                                    'optimizer_state_dict': q_optimizer.state_dict()}, model_path)
                        print('Best Validation Model saved to ', model_path)
                if args.use_scheduler:
                    q_scheduler.step(val_loss)

            if log_file:
                log_line += '\n'
                with open(log_file, 'a') as f:
                    f.write(log_line)
            if np.isnan(train_loss):
                print("Invalid value. Stopping training.")
                break



        print(min(train_errs))
    
    except KeyboardInterrupt:
        print('-' * 60)
        print('Early terminate')
    return model_path