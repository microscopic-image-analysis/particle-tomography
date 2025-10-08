import torch
from torch.utils.data import DataLoader

from .config import TrainingStep, GMMRejuvenateStep, SaveImagesStep
from .logging import save_output


class ParticleTomographyTrainer:
    def __init__(self, model, training_steps):
        self.optimizer = None       # a new optimizer is used for every training step. configured inside fit()
        self.geom_start_frac = None # set when configuring optimizer
        self.model = model
        self.training_steps = training_steps


    def configure_optimizer(self, lr=0.05, geom_opt_frac=1.0):
        self.geom_start_frac = geom_opt_frac
        volume_params = [
            self.model.points,
            self.model.log_point_weights,
            self.model.log_scale,
            self.model.log_noise_std,
            self.model.log_bandwidth
        ]
        geometry_params = [
            self.model.rotation_quats,
            self.model.shifts
        ]

        self.optimizer = torch.optim.Adam([
            {'params': volume_params, 'lr': lr},
            {'params': geometry_params, 'lr': 0.0}  # Start with 0 learning rate
        ])

    def calculate_r_factor(self, batch_size):
        """
        Calculate R-factor: R = sum(|I_obs - I_calc|) / sum(|I_obs|)
        where I_obs is the target and I_calc is the projected/predicted values
        """

        num_images = self.model.images.shape[0]
        loader = DataLoader(
            torch.arange(num_images),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True
        )

        total_abs_diff = 0.0
        total_abs_obs = 0.0

        self.model.eval()  # Set to evaluation mode
        with torch.no_grad():
            for batch_indices in loader:
                batch_indices = batch_indices.to(self.model.device, non_blocking=True)

                projected, target = self.model(batch_indices)

                # Calculate absolute differences and absolute observed values
                abs_diff = torch.abs(projected - target)
                abs_obs = torch.abs(target)

                total_abs_diff += abs_diff.sum().item()
                total_abs_obs += abs_obs.sum().item()

        self.model.train()  # Set back to training mode

        # Calculate R-factor
        r_factor = total_abs_diff / total_abs_obs if total_abs_obs > 0 else float('inf')
        return r_factor

    def train_step(self, batch_size, num_epochs):
        """Train for a given number of epochs with specified batch size"""
        num_images = self.model.images.shape[0]
        loader = DataLoader(
            torch.arange(num_images),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True
        )

        geometry_start_epoch = int(num_epochs * self.geom_start_frac)
        geometry_activated = False

        for epoch in range(num_epochs):
            if not geometry_activated and epoch >= geometry_start_epoch:
                # Activate geometry parameter optimization
                self.optimizer.param_groups[1]['lr'] = self.optimizer.param_groups[0]['lr']
                geometry_activated = True

            total_loss = 0.0
            for batch_indices in loader:
                batch_indices = batch_indices.to(self.model.device, non_blocking=True)

                self.optimizer.zero_grad()
                projected, target = self.model(batch_indices)
                loss = self.model.loss(projected, target)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * batch_indices.size(0)

            if epoch == num_epochs - 1:
                avg_loss = total_loss / num_images
                r_factor = self.calculate_r_factor(batch_size)

                print(f"Epoch {epoch + 1}/{num_epochs}, Loss per image: {avg_loss:.4f}")
                print(f"  Noise Std: {self.model.noise_std.item():.4f}")
                print(f"  R-factor: {r_factor:.4f}")
                print(f"  Current bandwidth: {self.model.bandwidth.item()}")


    def fit(self):
        """Run all training and rejuvenation steps"""
        for step_num, step in enumerate(self.training_steps):
            if isinstance(step, TrainingStep):
                self.configure_optimizer(
                    lr=step.learn_rate,
                    geom_opt_frac=step.geom_start_fraction,
                )
                self.train_step(batch_size=step.batch_size, num_epochs=step.n_iterations)

            elif isinstance(step, GMMRejuvenateStep):
                self.model.rejuvenate_GMM(step.rejuv_in_box)

            elif isinstance(step, SaveImagesStep):
                # save model
                self.model.save_model(path=step.out_dir)

                # Get current reconstruction
                volume = self.model.get_volume()
                true_volume = step.load_true_volume()

                # Save volume and images
                save_output(
                    reconstruction=volume,
                    true_volume=true_volume,
                    logging_prefix=step.logging_prefix,
                    outdir=step.out_dir,
                    slice_thickness=step.slice_thickness
                )


    def get_model(self):
        """Return the learned parameters"""
        return self.model
