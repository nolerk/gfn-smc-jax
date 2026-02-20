"""
Unified logging interface supporting both wandb and comet_ml.
"""

from typing import Any, Dict, List, Optional, Union
import matplotlib.pyplot as plt


class Logger:
    """Base logger class with no-op methods."""
    
    def init(self, config: Dict[str, Any], **kwargs) -> None:
        """Initialize the logger."""
        pass
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics."""
        pass
    
    def log_image(self, key: str, image: Any, step: Optional[int] = None) -> None:
        """Log an image."""
        pass
    
    def log_images(self, key: str, images: List[Any], step: Optional[int] = None) -> None:
        """Log multiple images."""
        pass
    
    def log_code(self, path: str = ".") -> None:
        """Log code files."""
        pass
    
    def set_summary(self, key: str, value: Any) -> None:
        """Set a summary value."""
        pass
    
    def finish(self, exit_code: int = 0) -> None:
        """Finish the logging session."""
        pass


class WandbLogger(Logger):
    """Weights & Biases logger."""
    
    def __init__(self):
        self.run = None
    
    def init(self, config: Dict[str, Any], **kwargs) -> None:
        import wandb
        # Handle log_config -> config mapping for wandb
        if 'log_config' in kwargs:
            kwargs['config'] = kwargs.pop('log_config')
        self.run = wandb.init(**config, **kwargs)
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        import wandb
        if step is not None:
            wandb.log(metrics, step=step)
        else:
            wandb.log(metrics)
    
    def log_image(self, key: str, image: Any, step: Optional[int] = None) -> None:
        import wandb
        wandb.log({key: wandb.Image(image)}, step=step)
    
    def log_images(self, key: str, images: List[Any], step: Optional[int] = None) -> None:
        import wandb
        wandb.log({key: [wandb.Image(img) for img in images]}, step=step)
    
    def log_code(self, path: str = ".") -> None:
        import wandb
        wandb.run.log_code(path)
    
    def set_summary(self, key: str, value: Any) -> None:
        import wandb
        wandb.run.summary[key] = value
    
    def finish(self, exit_code: int = 0) -> None:
        import wandb
        wandb.finish(exit_code=exit_code)


class CometLogger(Logger):
    """Comet.ml logger."""
    
    def __init__(self):
        self.experiment = None
    
    def init(self, config: Dict[str, Any], **kwargs) -> None:
        import comet_ml
        # Use comet-specific config if available, otherwise map from wandb config
        comet_config = {}
        
        if 'comet_config' in kwargs and kwargs['comet_config']:
            # Use comet-specific configuration
            comet_cfg = kwargs['comet_config']
            if 'project_name' in comet_cfg:
                comet_config['project_name'] = comet_cfg['project_name']
            if 'workspace' in comet_cfg and comet_cfg['workspace']:
                comet_config['workspace'] = comet_cfg['workspace']
            if 'experiment_name' in comet_cfg and comet_cfg['experiment_name']:
                comet_config['experiment_name'] = comet_cfg['experiment_name']
            if 'tags' in comet_cfg and comet_cfg['tags']:
                comet_config['tags'] = comet_cfg['tags'] if isinstance(comet_cfg['tags'], list) else [comet_cfg['tags']]
        else:
            # Map wandb config keys to comet_ml equivalents for backward compatibility
            if 'project' in config:
                comet_config['project_name'] = config['project']
            if 'entity' in config and config['entity']:
                comet_config['workspace'] = config['entity']
            if 'name' in config and config['name']:
                comet_config['experiment_name'] = config['name']
            if 'tags' in config and config['tags']:
                comet_config['tags'] = config['tags'] if isinstance(config['tags'], list) else [config['tags']]
            if 'mode' in config:
                if config['mode'] == 'disabled':
                    comet_config['disabled'] = True
        
        # Add any additional kwargs (like config dict for parameters)
        if 'log_config' in kwargs:
            comet_config['log_code'] = False  # We'll log code separately
        
        self.experiment = comet_ml.Experiment(**comet_config)
        
        # Log config as parameters if provided
        if 'log_config' in kwargs:
            self.experiment.log_parameters(kwargs['log_config'])
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        for key, value in metrics.items():
            if value is not None:
                self.experiment.log_metric(key, value, step=step)
    
    def log_image(self, key: str, image: Any, step: Optional[int] = None) -> None:
        # Handle different image types
        if isinstance(image, plt.Figure):
            self.experiment.log_figure(figure=image, figure_name=key, step=step)
        else:
            self.experiment.log_image(image, name=key, step=step)
    
    def log_images(self, key: str, images: List[Any], step: Optional[int] = None) -> None:
        for i, img in enumerate(images):
            if isinstance(img, plt.Figure):
                self.experiment.log_figure(figure=img, figure_name=f"{key}_{i}", step=step)
            else:
                self.experiment.log_image(img, name=f"{key}_{i}", step=step)
    
    def log_code(self, path: str = ".") -> None:
        self.experiment.log_code(folder=path)
    
    def set_summary(self, key: str, value: Any) -> None:
        # Comet doesn't have a direct summary equivalent, use log_other
        if value is not None:
            self.experiment.log_other(key, value)
    
    def finish(self, exit_code: int = 0) -> None:
        if exit_code != 0:
            self.experiment.log_other("error", True)
        self.experiment.end()


class MultiLogger(Logger):
    """Logger that supports multiple backends simultaneously."""
    
    def __init__(self, loggers: List[Logger]):
        self.loggers = loggers
    
    def init(self, config: Dict[str, Any], **kwargs) -> None:
        for logger in self.loggers:
            logger.init(config, **kwargs)
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        for logger in self.loggers:
            logger.log(metrics, step=step)
    
    def log_image(self, key: str, image: Any, step: Optional[int] = None) -> None:
        for logger in self.loggers:
            logger.log_image(key, image, step=step)
    
    def log_images(self, key: str, images: List[Any], step: Optional[int] = None) -> None:
        for logger in self.loggers:
            logger.log_images(key, images, step=step)
    
    def log_code(self, path: str = ".") -> None:
        for logger in self.loggers:
            logger.log_code(path)
    
    def set_summary(self, key: str, value: Any) -> None:
        for logger in self.loggers:
            logger.set_summary(key, value)
    
    def finish(self, exit_code: int = 0) -> None:
        for logger in self.loggers:
            logger.finish(exit_code=exit_code)


def create_logger(cfg) -> Logger:
    """Create a logger based on configuration.
    
    Args:
        cfg: Hydra configuration object
        
    Returns:
        Logger instance (single logger or multi-logger)
    """
    loggers = []
    
    # Check if wandb is enabled
    if hasattr(cfg, 'use_wandb') and cfg.use_wandb:
        loggers.append(WandbLogger())
    elif hasattr(cfg, 'logger') and cfg.logger.get('use_wandb', False):
        loggers.append(WandbLogger())
    
    # Check if comet_ml is enabled
    if hasattr(cfg, 'logger') and cfg.logger.get('use_comet', False):
        loggers.append(CometLogger())
    
    # Return appropriate logger
    if len(loggers) == 0:
        return Logger()  # No-op logger
    elif len(loggers) == 1:
        return loggers[0]
    else:
        return MultiLogger(loggers)


def init_logger(cfg, config_dict: Optional[Dict[str, Any]] = None) -> Logger:
    """Initialize and return a logger based on configuration.
    
    Args:
        cfg: Hydra configuration object
        config_dict: Optional config dictionary to log as parameters
        
    Returns:
        Initialized logger instance
    """
    logger = create_logger(cfg)
    
    # Prepare wandb config if present
    wandb_config = {}
    if hasattr(cfg, 'wandb'):
        wandb_config = dict(cfg.wandb)
    
    # Prepare comet config if present
    comet_config = None
    if hasattr(cfg, 'logger') and hasattr(cfg.logger, 'comet'):
        comet_config = dict(cfg.logger.comet)
    
    # Prepare init kwargs
    init_kwargs = {}
    if config_dict is not None:
        init_kwargs['log_config'] = config_dict  # Renamed to avoid conflict with 'config' parameter
    if comet_config is not None:
        init_kwargs['comet_config'] = comet_config
    
    logger.init(wandb_config, **init_kwargs)
    
    return logger


# Global logger instance for convenience
_logger: Optional[Logger] = None


def get_logger() -> Logger:
    """Get the global logger instance."""
    global _logger
    if _logger is None:
        _logger = Logger()  # No-op logger
    return _logger


def set_logger(logger: Logger) -> None:
    """Set the global logger instance."""
    global _logger
    _logger = logger


# Convenience functions that use the global logger
def log(metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    """Log metrics using the global logger."""
    get_logger().log(metrics, step=step)


def log_image(key: str, image: Any, step: Optional[int] = None) -> None:
    """Log an image using the global logger."""
    get_logger().log_image(key, image, step=step)


def log_images(key: str, images: List[Any], step: Optional[int] = None) -> None:
    """Log multiple images using the global logger."""
    get_logger().log_images(key, images, step=step)


def log_code(path: str = ".") -> None:
    """Log code using the global logger."""
    get_logger().log_code(path)


def set_summary(key: str, value: Any) -> None:
    """Set a summary value using the global logger."""
    get_logger().set_summary(key, value)


def finish(exit_code: int = 0) -> None:
    """Finish the global logger session."""
    get_logger().finish(exit_code=exit_code)