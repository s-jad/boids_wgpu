use bytemuck::Pod;
use futures::executor::block_on;
use nannou::{
    event::Update,
    wgpu::{self, Buffer, Queue},
    App,
};

use crate::{
    keyboard_controls::print_gpu_data, ui::update_ui, Boid, BoidParams, Model, PredatorParams,
    ViewParams, NUM_BOIDS, NUM_PREDATORS,
};

pub(crate) fn update(a: &App, m: &mut Model, u: Update) {
    update_predator_props(a, m, u);
    update_boid_props(a, m, u);
    update_cpu_read_buffers(a, m, u);
    update_ui::<[u32; NUM_PREDATORS]>(a, m, u);
}

pub(crate) fn update_view_params_buffer(app: &App, model: &mut Model) {
    let window = app.window(model.main_wid).unwrap();
    let queue = window.queue();

    let new_view_params = ViewParams {
        x_shift: model.variables.view_params.x_shift,
        y_shift: model.variables.view_params.y_shift,
        zoom: model.variables.view_params.zoom,
        time_modifier: model.variables.view_params.time_modifier,
    };

    queue.write_buffer(
        &model.variables.view_params_storage,
        0,
        bytemuck::cast_slice(&[new_view_params]),
    );
}

pub(crate) fn update_boid_params_buffer(app: &App, model: &mut Model) {
    let window = app.window(model.main_wid).unwrap();
    let queue = window.queue();

    let new_boid_params = BoidParams {
        max_velocity: model.variables.boid_params.max_velocity,
        min_velocity: model.variables.boid_params.min_velocity,
        turn_factor: model.variables.boid_params.turn_factor,
        visual_range: model.variables.boid_params.visual_range,
        protected_range: model.variables.boid_params.protected_range,
        centering_factor: model.variables.boid_params.centering_factor,
        self_avoid_factor: model.variables.boid_params.self_avoid_factor,
        predator_avoid_factor: model.variables.boid_params.predator_avoid_factor,
        matching_factor: model.variables.boid_params.matching_factor,
    };

    queue.write_buffer(
        &model.variables.boid_params_storage,
        0,
        bytemuck::cast_slice(&[new_boid_params]),
    );
}

pub(crate) fn update_predator_params_buffer(app: &App, model: &mut Model) {
    let window = app.window(model.main_wid).unwrap();
    let queue = window.queue();

    let new_pred_params = PredatorParams {
        max_velocity: model.variables.predator_params.max_velocity,
        min_velocity: model.variables.predator_params.min_velocity,
        turn_factor: model.variables.predator_params.turn_factor,
        pursuit_factor: model.variables.predator_params.pursuit_factor,
        pursuit_multiplier: model.variables.predator_params.pursuit_multiplier,
        matching_factor: model.variables.predator_params.matching_factor,
        self_avoid_factor: model.variables.predator_params.self_avoid_factor,
        visual_range: model.variables.predator_params.visual_range,
        protected_range: model.variables.predator_params.protected_range,
        interest_range: model.variables.predator_params.interest_range,
    };

    queue.write_buffer(
        &model.variables.predator_params_storage,
        0,
        bytemuck::cast_slice(&[new_pred_params]),
    );
}

pub(crate) fn update_cpu_read_buffers(app: &App, model: &mut Model, _update: Update) {
    let mw = app.window(model.main_wid).unwrap();
    let device = mw.device();
    let queue = mw.queue();

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("update_cpu_read_buffers encoder"),
    });

    encoder.copy_buffer_to_buffer(
        &model.buffers.boids_pos_buf,
        0,
        &model.buffers.cpu_read_boids_pos_buf,
        0,
        (std::mem::size_of::<[Boid; NUM_BOIDS]>()) as wgpu::BufferAddress,
    );

    encoder.copy_buffer_to_buffer(
        &model.buffers.predator_pos_buf,
        0,
        &model.buffers.cpu_read_predators_pos_buf,
        0,
        (std::mem::size_of::<[Boid; NUM_PREDATORS]>()) as wgpu::BufferAddress,
    );

    encoder.copy_buffer_to_buffer(
        &model.buffers.pursuits_buf,
        0,
        &model.buffers.cpu_read_predators_pursuits_buf,
        0,
        (std::mem::size_of::<[u32; NUM_PREDATORS]>()) as wgpu::BufferAddress,
    );

    encoder.copy_buffer_to_buffer(
        &model.buffers.captures_buf,
        0,
        &model.buffers.cpu_read_predators_captures_buf,
        0,
        (std::mem::size_of::<[u32; NUM_PREDATORS]>()) as wgpu::BufferAddress,
    );

    queue.submit(Some(encoder.finish()));
}

pub(crate) fn update_boid_props(app: &App, model: &mut Model, _update: Update) {
    let mw = app.window(model.main_wid).unwrap();
    let device = mw.device();
    let queue = mw.queue();

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("update_boid_props encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Boids SAC Compute Pass"),
        });
        compute_pass.set_pipeline(&model.pipelines.compute_sac_pipeline);
        compute_pass.set_bind_group(0, &model.compute_bind_group, &[]);
        compute_pass.set_bind_group(1, &model.uniforms.uniform_bind_group, &[]);
        compute_pass.dispatch_workgroups(16, 16, 1); // Adjust workgroup size as needed
    }

    queue.submit(Some(encoder.finish()));
}

pub(crate) fn update_predator_props(app: &App, model: &mut Model, _update: Update) {
    let mw = app.window(model.main_wid).unwrap();
    let device = mw.device();
    let queue = mw.queue();

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("update_predator_props encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Predator Compute Pass"),
        });
        compute_pass.set_pipeline(&model.pipelines.compute_pursuit_curve_pipeline);
        compute_pass.set_bind_group(0, &model.compute_bind_group, &[]);
        compute_pass.set_bind_group(1, &model.uniforms.uniform_bind_group, &[]);
        compute_pass.dispatch_workgroups(4, 1, 1); // Adjust workgroup size as needed
    }

    // println!("computing new predator pos!");
    queue.submit(Some(encoder.finish()));
}
