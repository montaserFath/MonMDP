"""Plotting functions for the training/testing, heatmaps for the q-tables, policy actions, and final trajectories"""
import colorsys
import seaborn as sns


def set_blind_colors() -> (list, list):
    """Get Blind colors friendly colors"""
    dark_hues = [0, 0.1, 0.4, 0.55, 0.65, 0.75, 0.9]
    dark_lightness = [0.4, 0.2, 0.15, 0.4, 0.4, 0.4, 0.4]
    light_hues = [1, 0.1, 0.3, 0.5, 0.6, 0.75, 0.85]
    light_lightness = [0.75, 0.6, 0.45, 0.6, 0.65, 0.8, 0.55]
    dark_colors = [colorsys.hls_to_rgb(h, dark_lightness[i], 1) for i, h in enumerate(dark_hues)]
    sns.palplot(dark_colors)
    light_colors = [colorsys.hls_to_rgb(h, light_lightness[i], 1) for i, h in enumerate(light_hues)]
    sns.palplot(light_colors)
    return light_colors, dark_colors
