"""System monitoring and health checks."""

import asyncio
import os
import psutil
import time
from typing import Dict, Optional

from src.core.logging import get_logger, performance_logger

logger = get_logger(__name__)


class SystemMonitor:
    """Monitor system resources and health."""
    
    def __init__(self):
        """Initialize system monitor."""
        self.process = psutil.Process(os.getpid())
        self._monitoring = False
        self._monitoring_task = None
    
    def start_monitoring(self, interval: int = 60) -> None:
        """Start periodic resource monitoring.
        
        Args:
            interval: Monitoring interval in seconds
        """
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitoring_task = asyncio.create_task(self._monitor_loop(interval))
        logger.info("system_monitoring_started", interval=interval)
    
    def stop_monitoring(self) -> None:
        """Stop periodic resource monitoring."""
        if not self._monitoring:
            return
        
        self._monitoring = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
        logger.info("system_monitoring_stopped")
    
    async def _monitor_loop(self, interval: int) -> None:
        """Background monitoring loop."""
        while self._monitoring:
            try:
                metrics = self.get_current_metrics()
                self._log_metrics(metrics)
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("monitoring_error", error=str(e))
                await asyncio.sleep(interval)
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current system metrics.
        
        Returns:
            Dictionary of metric name to value
        """
        # CPU metrics
        cpu_percent = self.process.cpu_percent(interval=0.1)
        
        # Memory metrics
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        virtual_memory = psutil.virtual_memory()
        system_memory_mb = virtual_memory.used / 1024 / 1024
        system_memory_available_mb = virtual_memory.available / 1024 / 1024
        
        metrics = {
            "process_cpu_percent": cpu_percent,
            "process_memory_mb": round(memory_mb, 2),
            "system_memory_used_mb": round(system_memory_mb, 2),
            "system_memory_available_mb": round(system_memory_available_mb, 2),
            "system_memory_percent": virtual_memory.percent,
        }
        
        # GPU metrics (if available)
        gpu_metrics = self._get_gpu_metrics()
        if gpu_metrics:
            metrics.update(gpu_metrics)
        
        return metrics
    
    def _get_gpu_metrics(self) -> Optional[Dict[str, float]]:
        """Get GPU metrics if available.
        
        Returns:
            Dictionary of GPU metrics or None
        """
        try:
            import pynvml
            
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            if device_count == 0:
                return None
            
            # Get metrics for the first GPU (or configured GPU)
            gpu_index = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
            if gpu_index >= device_count:
                gpu_index = 0
            
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            
            # Memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_memory_used_mb = mem_info.used / 1024 / 1024
            gpu_memory_total_mb = mem_info.total / 1024 / 1024
            
            # Utilization
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            # Temperature
            temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            
            return {
                "gpu_memory_used_mb": round(gpu_memory_used_mb, 2),
                "gpu_memory_total_mb": round(gpu_memory_total_mb, 2),
                "gpu_memory_percent": round(mem_info.used / mem_info.total * 100, 2),
                "gpu_utilization_percent": utilization.gpu,
                "gpu_temperature_c": temperature,
            }
            
        except Exception as e:
            logger.debug("gpu_metrics_unavailable", error=str(e))
            return None
        finally:
            try:
                pynvml.nvmlShutdown()
            except:
                pass
    
    def _log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log system metrics."""
        # Log memory usage
        performance_logger.log_memory_usage(
            used_mb=metrics["system_memory_used_mb"],
            available_mb=metrics["system_memory_available_mb"],
            gpu_used_mb=metrics.get("gpu_memory_used_mb"),
            gpu_total_mb=metrics.get("gpu_memory_total_mb"),
        )
        
        # Log general metrics
        logger.info(
            "system_metrics",
            cpu_percent=metrics["process_cpu_percent"],
            memory_mb=metrics["process_memory_mb"],
            gpu_utilization=metrics.get("gpu_utilization_percent"),
            gpu_temperature=metrics.get("gpu_temperature_c"),
        )
    
    def check_health(self) -> Dict[str, any]:
        """Perform health checks.
        
        Returns:
            Health check results
        """
        metrics = self.get_current_metrics()
        
        # Define health thresholds
        warnings = []
        if metrics["process_cpu_percent"] > 80:
            warnings.append("High CPU usage")
        
        if metrics["system_memory_percent"] > 90:
            warnings.append("High memory usage")
        
        if metrics.get("gpu_memory_percent", 0) > 90:
            warnings.append("High GPU memory usage")
        
        if metrics.get("gpu_temperature_c", 0) > 80:
            warnings.append("High GPU temperature")
        
        return {
            "status": "degraded" if warnings else "healthy",
            "warnings": warnings,
            "metrics": metrics,
        }


# Global system monitor instance
system_monitor = SystemMonitor()