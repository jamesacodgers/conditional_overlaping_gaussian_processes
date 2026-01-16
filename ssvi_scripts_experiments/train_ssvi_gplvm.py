import time
import torch
from src.ssvi_metrics import SSVIMetricsComputer
from src.ssvi_models_am_inf_sep_opt import GPLVM_SSVI_AM_INF
from src.ssvi_utils import SSVIResultsLogger

class SSVI_GPLVM(GPLVM_SSVI_AM_INF):
    """MNIST/FashionMNIST SSVI GP-LVM with Amortized Inference"""

    def __init__(self, config: OODSSVIConfig, Vx, mu_x, M):
        # Use init_lengthscales to create beta values
        beta = [[config.init_lengthscales] * config.latent_dim] * 1  # C=1 for GP-LVM

        # Use init_output_noise_scale for sigma2 initialization
        sigma2 = config.init_output_noise_scale

        # Initialize underlying SSVI model
        super().__init__(
            Vx=Vx,
            M=M,  # Observation dimension
            mu_x=mu_x,
            sigma2=sigma2,
            beta=beta,
            sigma2_s=config.sigma2_s,
            P=config.P,
            trainable_V=config.trainable_V,
            encoder_hidden_dims=config.encoder_hidden_dims,
            local_lr=config.local_lr,
            global_lr=config.global_lr,
            device=config.device,
            enable_missing_data=config.enable_missing_data,
        )

        # Store config
        self.config = config

    def train_step(
        self,
        Y_batch: torch.Tensor,
        prior_weight,
        U_samples: torch.Tensor = None,
    ):
        """
        Modified train_step: proper SSVI separation of local/global optimization
        """
        # Zero gradients for local optimizer (encoder) only initially
        self.optimizer_local.zero_grad()

        # Sample U before amortized inference step
        epsilon = self.get_epsilon()
        U_samples = self.get_u_samples(epsilon)

        # Handle missing data preprocessing for encoder input
        if self.enable_missing_data:
            # Replace NaN values with -9.0 for encoder input (encoder expects clean data)
            Y_batch_clean = torch.where(
                torch.isnan(Y_batch),
                torch.full_like(Y_batch, -9.0, dtype=torch.float64),
                Y_batch,
            )
            # Get local parameters via amortized inference (gradients only to encoder)
            mu_n_batch, sigma_n_batch = self.get_mu_sigma_x_train_samples(
                Y_batch_clean, epsilon
            )
        else:
            # Standard case: no missing data preprocessing needed
            mu_n_batch, sigma_n_batch = self.get_mu_sigma_x_train_samples(
                Y_batch, epsilon
            )

        # Now zero gradients for global optimizer before global computation
        self.optimizer_global.zero_grad()

        # Compute global ELBO loss (gradients flow to all parameters)
        if self.enable_missing_data:
            # Use missing data aware ELBO (coordinator handles the missing mask automatically)
            missing_mask = torch.isnan(Y_batch)
            # Use the same clean data for ELBO computation to maintain consistency with encoder
            loss = -self.ssvi_elbo(
                Y_batch_clean,
                mu_n_batch,
                sigma_n_batch,
                U_samples=U_samples,
                batch_size=self.config.batch_size,
                total_size=self.config.total_size,
                missing_mask=missing_mask,
                prior_weight=prior_weight
            )
        else:
            # Standard ELBO computation
            loss = -self.ssvi_elbo(
                Y_batch,
                mu_n_batch,
                sigma_n_batch,
                U_samples=U_samples,
                batch_size=self.config.batch_size,
                total_size=self.config.total_size,
                missing_mask=None,
                prior_weight=prior_weight
            )

        # Backward pass (gradients computed for all parameters)
        loss.backward()

        # Update parameters with different learning rates
        self.optimizer_local.step()
        self.optimizer_global.step()

        return loss.item()

    @property
    def mu_x(self):
        return self.mu_x_train

    @property
    def Sigma_x(self):
        return self.Sigma_x_train

    @torch.no_grad()
    def compute_full_elbo(
        self, y_full: torch.Tensor, missing_mask: torch.Tensor = None
    ) -> float:
        """
        Compute ELBO on full dataset using minibatches (for metrics and plots, not optimization).
        Processes data in minibatches to reduce memory usage, using the same parameters as training.

        Args:
            y_full: Full dataset [N, M]
            missing_mask: Optional boolean mask for missing data [N, M]

        Returns:
            Full dataset ELBO value (scalar)
        """
        # Sample U for ELBO computation (same U for all batches)
        epsilon = self.get_epsilon()
        U_samples = self.get_u_samples(epsilon)

        # Use same batch size as training
        batch_size = self.config.batch_size
        N = y_full.shape[0]

        # Process data in minibatches and collect ELBOs (same as training loop)
        batch_elbos = []

        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            y_batch = y_full[start_idx:end_idx]

            # Handle missing data preprocessing for encoder input (same as training)
            if self.enable_missing_data and missing_mask is not None:
                mask_batch = missing_mask[start_idx:end_idx]
                y_batch_clean = torch.where(
                    torch.isnan(y_batch),
                    torch.full_like(y_batch, -9.0, dtype=torch.float64),
                    y_batch,
                )
            else:
                y_batch_clean = y_batch
                mask_batch = None

            # Get latent parameters via amortized inference for this batch
            mu_n_batch, sigma_n_batch = self.get_mu_sigma_x_train_samples(
                y_batch_clean, epsilon
            )

            # Compute ELBO for this batch (EXACTLY same as in training: batch_size=config.batch_size, total_size=config.total_size)
            if self.enable_missing_data and mask_batch is not None:
                batch_elbo = self.ssvi_elbo(
                    y_batch_clean,
                    mu_n_batch,
                    sigma_n_batch,
                    U_samples,
                    batch_size=self.config.batch_size,
                    total_size=self.config.total_size,
                    missing_mask=mask_batch,
                )
            else:
                batch_elbo = self.ssvi_elbo(
                    y_batch_clean,
                    mu_n_batch,
                    sigma_n_batch,
                    U_samples,
                    batch_size=self.config.batch_size,
                    total_size=self.config.total_size,
                    missing_mask=None,
                )

            batch_elbos.append(batch_elbo)

        # Average the batch ELBOs (same approach as would be used in training)
        mean_elbo = sum(batch_elbos) / len(batch_elbos)

        return mean_elbo.item()
    


def train_ssvi(
    model: SSVI_GPLVM,
    y_train,
    config: OODSSVIConfig,
    logger: SSVIResultsLogger = None,
    split_data=None,
    labels_train=None,
):
    """Training function for SSVI GP-LVM with optional logging and metrics"""
    losses = []  # Batch ELBO losses (for training monitoring)
    full_elbo_losses = []  # Full dataset ELBO (for metrics and plots)
    full_elbo_iterations = []  # Iterations at which full ELBO was computed

    print(f"Training {config.id_dataset} SSVI GP-LVM...")
    print(f"Iterations: {config.training_iterations}, Batch size: {config.batch_size}")

    # Initialize metrics computer if enabled
    metrics_computer = None
    if config.enable_metrics and split_data is not None:
        print(
            f"Metrics computation enabled (frequency: every {config.metrics_frequency} iterations)"
        )
        metrics_computer = SSVIMetricsComputer(
            device=config.device, metrics_batch_size=config.batch_size
        )

    # Initialize k-NN metrics tracking
    knn_metrics_history = []
    knn_metrics_iterations = []

    start_time = time.time()
    init_prior_weight=10
    for i in range(config.training_iterations):
        # Sample batch indices
        N = y_train.shape[0]
        batch_indices = torch.randperm(N, device=y_train.device)[: config.batch_size]
        y_batch = y_train[batch_indices]

        # Training step - U sampling happens inside automatically each step
        # prior_weight = init_prior_weight 
        # prior_weight = init_prior_weight + (i/config.training_iterations)*(1-init_prior_weight)
        prior_weight = (init_prior_weight-1)/2 + (init_prior_weight-1)/2*torch.cos(torch.tensor(torch.pi*i/config.training_iterations)) + 1
        loss = model.train_step(y_batch, batch_indices, prior_weight= prior_weight)
        losses.append(loss)

        # Metrics computation (periodic)
        if (
            metrics_computer is not None
            and config.enable_metrics
            and i % config.metrics_frequency == 0
        ):

            print(f"  Computing metrics at iteration {i}...")

            # Compute full dataset loss (negative ELBO, for metrics and plots)
            if config.enable_missing_data:
                full_elbo = model.compute_full_elbo(
                    split_data["y_train"], missing_mask=split_data["train_missing_mask"]
                )
            else:
                full_elbo = model.compute_full_elbo(split_data["y_train"])

            # Store as negative ELBO (loss) for consistency with batch losses
            full_elbo_losses.append(-full_elbo)
            full_elbo_iterations.append(i)
            print(f"    Full Dataset Loss: {-full_elbo:.4f} (batch loss: {loss:.4f})")

            # Use full datasets for proper evaluation
            # Compute metrics with missing data info if enabled
            if config.enable_missing_data:
                metrics = metrics_computer.compute_full_metrics(
                    model=model,
                    y_train=split_data["y_train"],
                    y_test=split_data["y_test"],
                    iteration=i,
                    elbo_loss=full_elbo,  # Use full dataset ELBO
                    n_samples=32,  # Fixed number of samples for metrics computation
                    y_train_original=split_data["y_train_original"],
                    y_test_original=split_data["y_test_original"],
                    train_missing_mask=split_data["train_missing_mask"],
                    test_missing_mask=split_data["test_missing_mask"],
                )
            else:
                metrics = metrics_computer.compute_full_metrics(
                    model=model,
                    y_train=split_data["y_train"],
                    y_test=split_data["y_test"],
                    iteration=i,
                    elbo_loss=full_elbo,  # Use full dataset ELBO
                    n_samples=32,  # Fixed number of samples for metrics computation
                )

            print(
                f"    Train RMSE: {metrics.train_rmse:.4f}, Test RMSE: {metrics.test_rmse:.4f} "
            )
            print(
                f"    Train NLPD: {metrics.train_nlpd:.3f}, Test NLPD: {metrics.test_nlpd:.3f}"
            )
            if config.enable_missing_data and not np.isnan(metrics.train_missing_nlpd):
                print(f"    Train Missing Data NLPD: {metrics.train_missing_nlpd:.3f}")
                print(f"    Test Missing Data NLPD: {metrics.test_missing_nlpd:.3f}")
            if config.enable_missing_data and not np.isnan(metrics.train_missing_rmse):
                print(f"    Train Missing Data RMSE: {metrics.train_missing_rmse:.4f}")
                print(f"    Test Missing Data RMSE: {metrics.test_missing_rmse:.4f}")
            print(
                f"    Effective dims: {metrics.effective_latent_dims}/{metrics.total_latent_dims}"
            )
            print(
                f"    Full datasets: Train={split_data['y_train'].shape[0]}, Test={split_data['y_test'].shape[0]} samples"
            )

            # Compute k-NN clustering quality metrics
            print(f"    Computing k-NN clustering metrics...")
            knn_results = compute_knn_accuracies_ssvi(
                model=model,
                y_train=split_data['y_test'],
                labels_train=split_data['labels_test'],
                k_values=[1, 3, 7, 12],
                batch_size=config.batch_size,
            )
            print(f"      k=1: {knn_results['knn_1']:.4f}")
            print(f"      k=3: {knn_results['knn_3']:.4f}")
            print(f"      k=7: {knn_results['knn_7']:.4f}")
            print(f"      k=12: {knn_results['knn_12']:.4f}")
            knn_metrics_history.append(knn_results)
            knn_metrics_iterations.append(i)

            # Log metrics
            if logger:
                logger.log_training(f"Metrics at iteration {i}: {metrics.to_dict()}")
                logger.log_training(f"k-NN metrics at iteration {i}: {knn_results}")

        # Checkpoint saving (models and plots together)
        if logger and i > 0 and i % config.checkpoint_frequency == 0:
            print(f"  Saving checkpoint at iteration {i}...")
            try:
                # Save checkpoint model
                checkpoint_filename = f"ood_ssvi_iter_{i}_seed_{config.random_seed}.pkl"
                logger.save_model(model, checkpoint_filename, is_checkpoint=True)
                print(f"    Checkpoint model saved: {checkpoint_filename}")

                # Save checkpoint plot (if enabled)
                if config.save_training_plots:
                    plot_filename = f"training_results_iter_{i}.png"
                    try:
                        from src.mnist_ood_plot_helper import visualize_ssvi_results

                        # Use full ELBO losses for checkpoint plots
                        visualize_ssvi_results(
                            model=model,
                            y_train=y_train,
                            labels_train=labels_train,
                            losses=full_elbo_losses,  # Use full dataset ELBO for plots
                            config=config,
                            logger=logger,
                            save_filename=plot_filename,
                            show_plot=False,
                            is_checkpoint=True,
                            train_indices=split_data["train_indices"],
                            dataset_name=config.id_dataset,
                            is_ood=False,
                            full_elbo_iterations=full_elbo_iterations,
                            batch_losses=losses  # Batch losses up to current iteration
                        )
                        print(f"    Checkpoint plot saved: {plot_filename}")
                    except Exception as e:
                        print(
                            f"    Warning: Could not create checkpoint visualization at iteration {i}: {e}"
                        )

                    # Save prior samples plots at checkpoint
                    try:
                        plots_checkpoints_dir = os.path.join(
                            logger.plots_dir, "checkpoints"
                        )
                        os.makedirs(plots_checkpoints_dir, exist_ok=True)

                        prior_samples_path = os.path.join(
                            plots_checkpoints_dir,
                            f"prior_samples_iter_{i}_{config.random_seed}.png",
                        )
                        visualize_prior_samples(
                            model=model,
                            n_images=24,
                            save_path=prior_samples_path,
                            show_plot=False,
                            dataset_name=config.id_dataset,
                            image_shape=(28, 28),
                            scaler=split_data.get("scaler"),
                        )
                        print(
                            f"    Prior samples checkpoint saved: prior_samples_iter_{i}_{config.random_seed}.png"
                        )

                        prior_samples_avg_path = os.path.join(
                            plots_checkpoints_dir,
                            f"prior_samples_averaged_iter_{i}_{config.random_seed}.png",
                        )
                        visualize_prior_samples_averaged(
                            model=model,
                            n_images=8,
                            save_path=prior_samples_avg_path,
                            show_plot=False,
                            dataset_name=config.id_dataset,
                            image_shape=(28, 28),
                            scaler=split_data.get("scaler"),
                        )
                        print(
                            f"    Prior samples averaged checkpoint saved: prior_samples_averaged_iter_{i}_{config.random_seed}.png"
                        )

                        test_image_predictions_checkpoint_path = os.path.join(plots_checkpoints_dir, f"test_image_predictions_iter_{i}_{config.random_seed}.png")
                        visualize_test_image_predictions(
                            model=model,
                            y_train=split_data['y_test'],
                            config=config,
                            n_images=24,
                            save_path=test_image_predictions_checkpoint_path,
                            show_plot=False,
                            dataset_name=config.id_dataset,
                            image_shape=(28, 28),
                            scaler=split_data.get('scaler')
                        )
                        print(f"    Test image predictions checkpoint saved: test_image_predictions_iter_{i}_{config.random_seed}.png")
                    except Exception as e:
                        print(
                            f"    Warning: Could not create prior samples checkpoint at iteration {i}: {e}"
                        )

            except Exception as e:
                print(f"    Warning: Error saving checkpoint at iteration {i}: {e}")

        # Gradient logging (only if logger provided and enabled)
        if (
            logger
            and config.enable_gradient_logging
            and i % config.gradient_log_frequency == 0
        ):
            gradient_info = model.get_gradient_info()
            logger.log_gradients(i, loss, gradient_info)

        # Progress reporting
        if i % config.log_frequency == 0:
            elapsed = time.time() - start_time
            message = f"Iteration {i}/{config.training_iterations}: Loss={loss:.4f}, Time={elapsed:.1f}s"
            print(f"Iteration {i}/{config.training_iterations}")
            print(f"  Loss: {loss:.4f}")
            print(f"  Time elapsed: {elapsed:.1f}s")
            # Log to file
            if logger:
                logger.log_training(message)

    total_time = time.time() - start_time

    # Compute final full dataset loss (negative ELBO)
    if config.enable_missing_data and split_data is not None:
        final_full_elbo = model.compute_full_elbo(
            split_data["y_train"], missing_mask=split_data["train_missing_mask"]
        )
    elif split_data is not None:
        final_full_elbo = model.compute_full_elbo(split_data["y_train"])
    else:
        if config.enable_missing_data:
            missing_mask = torch.isnan(y_train)
        else:
            missing_mask = None
        final_full_elbo = model.compute_full_elbo(y_train, missing_mask=missing_mask)

    # Store as negative ELBO (loss) for consistency with batch losses
    full_elbo_losses.append(-final_full_elbo)
    full_elbo_iterations.append(config.training_iterations)

    final_message = f"Training completed! Final Full Loss: {-final_full_elbo:.4f} (batch loss: {losses[-1]:.4f}), Total time: {total_time:.1f}s ({config.training_iterations/total_time:.1f} iter/s)"
    print(f"Training completed!")
    print(f"Final Full Dataset Loss: {-final_full_elbo:.4f}")
    print(f"Final Batch Loss: {losses[-1]:.4f}")
    print(
        f"Total time: {total_time:.1f}s ({config.training_iterations/total_time:.1f} iter/s)"
    )
    # Log to file
    if logger:
        logger.log_training(final_message)

    # Return full ELBO losses for plots (with iterations for x-axis)
    return {
        "batch_losses": losses,
        "full_elbo_losses": full_elbo_losses,
        "full_elbo_iterations": full_elbo_iterations,
        "knn_metrics_history": knn_metrics_history,
        "knn_metrics_iterations": knn_metrics_iterations,
    }, metrics_computer