use nannou::{
    glam::Vec2,
    prelude::WindowId,
    wgpu::{self, BindGroupLayout, PipelineLayout},
};
use nannou_egui::Egui;

pub(crate) struct Model {
    pub(crate) pipelines: Pipelines,
    pub(crate) compute_bind_group: wgpu::BindGroup,
    pub(crate) buffers: Buffers,
    pub(crate) uniforms: Uniforms,
    pub(crate) variables: Variables,
    pub(crate) controls: Controls,
    pub(crate) main_wid: WindowId,
    pub(crate) ui: Egui,
}

pub(crate) struct Pipelines {
    pub(crate) render_pipeline: wgpu::RenderPipeline,
    pub(crate) compute_boid_pos_pipeline: wgpu::ComputePipeline,
    pub(crate) compute_predator_pos_pipeline: wgpu::ComputePipeline,
    pub(crate) compute_sac_pipeline: wgpu::ComputePipeline,
    pub(crate) compute_pursuit_curve_pipeline: wgpu::ComputePipeline,
}

pub(crate) struct Uniforms {
    pub(crate) time_uniform: wgpu::Buffer,
    pub(crate) resolution_uniform: wgpu::Buffer,
    pub(crate) uniform_bind_group: wgpu::BindGroup,
}

pub(crate) struct Buffers {
    pub(crate) vertex_buf: wgpu::Buffer,
    pub(crate) boids_pos_buf: wgpu::Buffer,
    pub(crate) cpu_read_boids_pos_buf: wgpu::Buffer,
    pub(crate) predator_pos_buf: wgpu::Buffer,
    pub(crate) pursuits_buf: wgpu::Buffer,
    pub(crate) captures_buf: wgpu::Buffer,
    pub(crate) captured_boids_buf: wgpu::Buffer,
    pub(crate) cpu_read_predators_pos_buf: wgpu::Buffer,
    pub(crate) cpu_read_predators_pursuits_buf: wgpu::Buffer,
    pub(crate) cpu_read_predators_captures_buf: wgpu::Buffer,
}

pub(crate) struct Variables {
    pub(crate) view_params: ViewParams,
    pub(crate) view_params_storage: wgpu::Buffer,
    pub(crate) boid_params: BoidParams,
    pub(crate) boid_params_storage: wgpu::Buffer,
    pub(crate) predator_params: PredatorParams,
    pub(crate) predator_params_storage: wgpu::Buffer,
    pub(crate) variable_bind_group: wgpu::BindGroup,
}

#[derive(Debug)]
pub(crate) struct Controls {
    pub(crate) kcm: KeyboardControlMode,
}

pub(crate) type TimeUniform = f32;
pub(crate) type ResolutionUniform = Vec2;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct ViewParams {
    pub(crate) x_shift: f32,
    pub(crate) y_shift: f32,
    pub(crate) zoom: f32,
    pub(crate) time_modifier: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct Boid {
    pub(crate) pos: [f32; 2],
    pub(crate) vel: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct BoidParams {
    pub(crate) max_velocity: f32,
    pub(crate) min_velocity: f32,
    pub(crate) turn_factor: f32,
    pub(crate) visual_range: f32,
    pub(crate) protected_range: f32,
    pub(crate) centering_factor: f32,
    pub(crate) self_avoid_factor: f32,
    pub(crate) predator_avoid_factor: f32,
    pub(crate) matching_factor: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct PredatorParams {
    pub(crate) max_velocity: f32,
    pub(crate) min_velocity: f32,
    pub(crate) turn_factor: f32,
    pub(crate) pursuit_factor: f32,
    pub(crate) pursuit_multiplier: f32,
    pub(crate) matching_factor: f32,
    pub(crate) self_avoid_factor: f32,
    pub(crate) visual_range: f32,
    pub(crate) protected_range: f32,
    pub(crate) interest_range: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Vertex {
    pub(crate) position: [f32; 2],
}

pub(crate) const VERTICES: &[Vertex; 6] = &[
    // Bottom left triangle
    Vertex {
        position: [-1.0, -1.0],
    },
    Vertex {
        position: [1.0, -1.0],
    },
    Vertex {
        position: [-1.0, 1.0],
    },
    // Top right triangle
    Vertex {
        position: [1.0, -1.0],
    },
    Vertex {
        position: [1.0, 1.0],
    },
    Vertex {
        position: [-1.0, 1.0],
    },
];

pub(crate) struct Layouts {
    pub(crate) uniform_bind_group_layout: BindGroupLayout,
    pub(crate) variable_bind_group_layout: BindGroupLayout,
    pub(crate) compute_bind_group_layout: BindGroupLayout,
    pub(crate) render_pipeline_layout: PipelineLayout,
    pub(crate) compute_pipeline_layout: PipelineLayout,
}

// ENUMS
#[derive(Debug)]
pub(crate) enum KeyboardControlMode {
    View,
    Boids,
    Predator,
    Debug,
}
