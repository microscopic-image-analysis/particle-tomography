from particle_tomography.config import TrainingStep, TrainingConfig, GMMRejuvenateStep, SaveImagesStep


class TrainingPlan:
    """Ordered list of optimization, rejuvenation, and output steps."""

    def __init__(self):
        self.steps = []

    def add_training_step(self, n_iterations, batch_size, learn_rate, geom_start_fraction=1.0):
        """Append a gradient-optimization step and return this plan."""
        self.steps.append(TrainingStep(
            n_iterations=n_iterations,
            learn_rate=learn_rate,
            batch_size=batch_size,
            geom_start_fraction=geom_start_fraction
        ))
        return self  # allow chaining

    def add_gmm_rejuvenate(self, rejuv_in_box=True):
        """Append a Gaussian-mixture particle rejuvenation step and return this plan."""
        self.steps.append(GMMRejuvenateStep(rejuv_in_box=rejuv_in_box))
        return self

    def add_save_images(self, out_dir, slice_thickness=10, logging_prefix="final", true_volume_path=None, true_volume_loader=None):
        """Append a step that saves the current reconstruction and diagnostic images."""
        self.steps.append(SaveImagesStep(out_dir=out_dir, slice_thickness=slice_thickness, logging_prefix=logging_prefix,
                                         true_volume_path=true_volume_path, true_volume_loader=true_volume_loader))

    def get_steps(self):
        """Return the configured steps in execution order."""
        return self.steps


def build_simple_plan(total_iterations, batch_size, lr=2.5e-3,geom_start_fraction=0.9, num_rejuvenates=1, rejuv_in_box=True):
    """Build a default training plan with evenly spaced rejuvenation steps."""
    iterations = total_iterations // (num_rejuvenates + 1)
    plan = TrainingPlan()
    plan.add_training_step(iterations, batch_size, lr, geom_start_fraction)
    for i in range(num_rejuvenates):
        plan.add_gmm_rejuvenate(rejuv_in_box)
        plan.add_training_step(iterations, batch_size, lr, geom_start_fraction)
    return plan


def build_plan_from_config(config: TrainingConfig) -> TrainingPlan:
    """Convert a serializable TrainingConfig into an executable TrainingPlan."""
    plan = TrainingPlan()
    for step in config.steps:
        if isinstance(step, TrainingStep):
            plan.add_training_step(
                n_iterations=step.n_iterations,
                learn_rate=step.learn_rate,
                batch_size=step.batch_size,
                geom_start_fraction=step.geom_start_fraction,
            )
        elif isinstance(step, GMMRejuvenateStep):
            plan.add_gmm_rejuvenate(rejuv_in_box=step.rejuv_in_box)
        elif isinstance(step, SaveImagesStep):
            plan.add_save_images(out_dir=step.out_dir, slice_thickness=step.slice_thickness, logging_prefix=step.logging_prefix,
                                 true_volume_path=step.true_volume_path, true_volume_loader=step.true_volume_loader)
        else:
            raise ValueError(f"Unknown step type {type(step)}")
    return plan

