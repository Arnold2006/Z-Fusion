"""
System Monitor UI Component

Provides system monitor UI components that can be placed in each tab,
with a single shared timer in app.py to avoid multiple timer conflicts.

Usage:
1. In each tab module, call create_monitor_textboxes() to create the UI
2. Register the components with services.inter_module
3. In app.py, call setup_shared_monitor_timer() after all tabs are loaded
"""

import gradio as gr
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules import SharedServices


def create_monitor_textboxes() -> tuple[gr.Textbox, gr.Textbox]:
    """
    Create GPU and CPU monitor textboxes for placement in a tab.
    
    Returns:
        Tuple of (gpu_monitor, cpu_monitor) textboxes
        
    Example usage in a tab:
        with gr.Row():
            with gr.Column(scale=1, min_width=200):
                gpu_monitor, cpu_monitor = create_monitor_textboxes()
    """
    with gr.Row():
        with gr.Column(scale=1, min_width=200):
            gpu_monitor = gr.Textbox(
                value="Loading...",
                lines=4.5,
                container=False,
                interactive=False,
                show_label=False,
                elem_classes=["monitor-box", "gpu-monitor"]
            )
        with gr.Column(scale=1, min_width=200):
            cpu_monitor = gr.Textbox(
                value="Loading...",
                lines=4,
                container=False,
                interactive=False,
                show_label=False,
                elem_classes=["monitor-box", "cpu-monitor"]
            )
    
    return gpu_monitor, cpu_monitor


def setup_shared_monitor_timer(services: "SharedServices", interval: float = 2.0) -> gr.Timer:
    """
    Create a single shared timer that updates all registered monitor components.
    
    Call this in app.py AFTER all tabs have been created and have registered
    their monitor components via services.inter_module.
    
    Expected registered components (per tab):
        - {tab_id}_gpu_monitor
        - {tab_id}_cpu_monitor
    
    Args:
        services: SharedServices instance
        interval: Update interval in seconds (default 2.0)
        
    Returns:
        The gr.Timer instance
    """
    # Collect all registered monitor components
    monitor_outputs = []
    
    # Known tab IDs that have monitors
    tab_ids = ["zimage", "upscale", "experimental"]
    
    for tab_id in tab_ids:
        gpu = services.inter_module.get_component(f"{tab_id}_gpu_monitor")
        cpu = services.inter_module.get_component(f"{tab_id}_cpu_monitor")
        if gpu and cpu:
            monitor_outputs.extend([gpu, cpu])
    
    if not monitor_outputs:
        # No monitors registered, create a dummy timer that does nothing
        timer = gr.Timer(interval, active=False)
        return timer
    
    # Calculate how many tab pairs we have
    num_tabs = len(monitor_outputs) // 2
    
    def update_all_monitors():
        """Update all monitor displays with current system info."""
        try:
            if services.system_monitor:
                gpu_info, cpu_info = services.system_monitor.get_system_info()
            else:
                gpu_info, cpu_info = "N/A", "N/A"
        except Exception:
            gpu_info, cpu_info = "Error", "Error"
        
        # Return the same values for each tab's monitors
        # Output order: [gpu1, cpu1, gpu2, cpu2, gpu3, cpu3, ...]
        results = []
        for _ in range(num_tabs):
            results.extend([gpu_info, cpu_info])
        
        return results
    
    # Create single timer
    timer = gr.Timer(interval, active=True)
    timer.tick(fn=update_all_monitors, outputs=monitor_outputs)
    
    return timer
