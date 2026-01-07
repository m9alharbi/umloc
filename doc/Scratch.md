# What is done differently from the RONIN paper:
1. Gaussian filter to filter out the noise.
2. Velocity is used instead of displacement.
3. Outliers are removed for better training.
4. Data transformation.
5. Sensor calibration.
6. Frame transformation.

# Why is the behaviour different from other papers where the prediction has randomness compared to the ground truth?
1. when computing the velocity, I was not dividing by dt.
2. The unit of dt is different when computing the velocity and position (In my data collection process, I used milliseconds, whereas, in the papers, they used seconds).
3. The frequency of collecting the data is slow (30 Hz) compared to (200Hz).
4. Filtering the data and removing outliers.
5. The scaling of the plots.
6. In my case, the path is closed and repeated multiple times, which causes this behavior.

# The results without a map:
- Proposal results.
- Zed IMU results.
- Updated results (Dividing by dt, units change, Scaling the plot, one repetition)
- RoNIN results.
# Performance table:

| Sequences | Proposal | ZED | Updated | 1 round | 
|:---------:|:--------:|:---:|:-------:|:-------:|
| Seq_1 | (2.06, 1.13) | (0.98, 0.36) | (2.48, 0.89) | (0.92, 1.17) |
| Seq_2 | (1.14, 1.17) | (0.67, 0.31) | (1.32, 0.76) | (1.30, 1.17) |
| Seq_3 | (2.02, 2.44 | (0.56, 0.36) | (1.45, 0.83) | (1.22, 1.38) |
| Seq_4 | (1.54, 0.99) | (0.66, 0.35) | (1.33, 0.88) | (0.74, 0.95) |

# RoNIN
ATE: 1.63 & 2.00 & 1.66  
RTE: 1.91 & 2.64 & 2.16

***
To Do:
- [ ] Add a new dataloader for the RoNIN dataset
- [ ] Evaluate the model and the quantile model on the RoNIN

The trajectory loss is given by:

$\text{traj\_loss} = \frac{1}{N \cdot T} \sum_{i=1}^{N} \sum_{t=1}^{T} \left( \sum_{k=1}^{t} \left( \hat{\mathbf{y}}_{k,i} - \mathbf{y}_{k,i} \right)^2 \right)$

where:
- $\hat{\mathbf{y}}_{k,i} $ is the predicted value at time step $k$ for sample $i$.
- $\mathbf{y}_{k,i}$ is the target value at - [ ] Update the code on github by adding the branchtime step $k$ for sample $i$.
- $N$ is the total number of samples.
- $T$ is the total number of time steps.

# RoNIN Dataset
In the training:
1) Calibrated the acceleration and the gyroscope
2) Velocity calculation
3) Coordinate frame transformation from IMU frame to tango position frame
4) Time axis alignment (synchronization)
5) Get features, targets, time, orientation and ground truth position
6) Noise filtration with Gaussian 1D filter for IMU
7) Removing outliers and random shift the data
8) Shuffle the data
9) Get the batched features and targets
10) Apply random angle rotations on the global batched input and output
11) Used EKF in the orientation
In the testing:
- No shuffling for the data
- grv_only is set to true
- No random angle transformation (totally rely on the game rotation vector of the device)
- No random shift
- At test time, we use the coordinate frame defined by system device orientations from Android, whose Z axis is aligned with gravity
- Check the orientation there is something missing when using grv_only


grv_only = only use the game rotation vector for orientation

### 
1) Choose the dataset (RoNIN, ours).
2) Choose the target type.
3) Map choice (with, without).
4) Choose the model (single task, multiple tasks, quantile).
5) Choose Loss function (Trajectory loss, trajectory with penalty, trajectory with map loss, quantile loss).
6) Training process.


ConvLSTM hidden feature vector
CNN context feature vector
feasible area map

1. Decoder Input Design:
   1) Use the initial position as the input to the decoder (one-hot encoded or a feature representation).
   2) Incorporate the feasible area map into the decoder. This will guide the sampling to prioritize plausible locations
   3) Spatial Conditioning: Use spatial features of the feasible map as a conditioning layer or mask to bias the generated points towards valid regions.
   4) Mask Multiplication: For each step, you can multiply the decoder's output with the feasible area map at that step to ensure the sampled point is within a valid area.
   5) Gumbel-Softmax or Categorical Sampling: If feasible locations are sparse, a Gumbel-Softmax sampling mechanism or categorical sampling approach can be used to force the sampled position to lie in high-probability (high-mask value) regions.
   6) The feasible area map should be updated step-by-step, reflecting the current location and ensuring valid transitions between steps.
   7) The ConvLSTM can take the feasible area map and the sampled location at each step as inputs. The feature maps from the ConvLSTM will act as an attention mask or conditioning factor for sampling.

--------------------------------------------------------------------
Different Sampling Approaches:
You can incorporate a masked sampling approach to ensure the GAN respects the valid regions on the map and stays within the predicted bounds:

1) Masking the Latent Space: Before sampling the next position, apply the binary mask generated from the environment and quantile bounds to limit the sampling space. This ensures the GAN only samples positions in valid areas.
2) Conditional Sampling: Use the interpolated quantile bounds and the environment map as conditioning inputs during sampling. This can be done by using the binary mask as a part of the conditioning input, ensuring the sampled positions are valid.
3) Dynamic Weighting: Another approach is to modify the loss function to penalize the generator when it proposes locations outside the valid areas or bounds. You can give more weight to the penalty when the generated position is closer to obstacles or falls outside the quantiles.
    Fall outside the quantile bounds.
    Lie in invalid areas as per the environment map.
    Create abrupt, unrealistic jumps from the previous position (for smoothness).
4) Rejection Sampling: This was implemented, where points are generated and then filtered based on the Gaussian map. Invalid points are discarded, and only valid points are retained.
5) Importance Sampling: This approach biases the sampling process to favor regions where the Gaussian kernel map values are lower (i.e., valid regions).
6) Masked Sampling: During generation, you apply a mask that prevents the generator from producing points in invalid areas by adjusting the loss function or using a direct masking operation during the forward pass.
7) Masked Filtering during Training:
    The generator produces points as usual, but we apply a mask based on the Gaussian kernel map.
    Points that do not satisfy the threshold condition are removed from the training batch, and the loss is computed only for valid points.
    This enforces that the generator learns to produce valid points during training, rather than filtering them post-generation.
9) Attention Mechanisms: Integrate attention mechanisms to dynamically guide the generator towards valid regions of the output space.

-----------------------------------------------------------------------
5. Discriminator Design:
The discriminator should evaluate the entire generated trajectory, not just individual points:
Use a CNN-based architecture that takes in the trajectory and the environment map to output a probability score indicating whether the trajectory is plausible or not.
Include trajectory features such as smoothness, obstacle avoidance, and boundary adherence as evaluation metrics for training.

3. Generator Decoder Loss Design:
Feasible Map Consistency Loss:
Add a consistency loss that penalizes generated positions that fall outside the high-probability areas defined in the feasible area map.
Use a binary cross-entropy loss between the generated mask (based on the generated position) and the feasible area map.
Sequential Trajectory Loss:
Include a smoothness loss to ensure that the generated trajectory maintains spatial and temporal consistency.
Use a distance-based loss (e.g., L2 loss) to minimize sudden jumps between generated positions.

1) Find an efficient way for the CONVLSTM training 
2) Sampling method ==> Different approaches require stochastic methods to compute the gradient (policy gradient) and update the network design in addition to taking longer time for sampling
3) Loss function design ==> Include a differentiable map representation to update the network weights and consider the map factor
4) Attention mechanism ==> Do not use the attention not suitable for my application, not memory efficient not scalable with longer sequences
5) Map design ==> Check with GPT for a differentiable map or other representation
6) Other datasets ==> Find a way to get the RoNIN map or look for another dataset