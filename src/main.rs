mod misc;
use ui::*;
mod ui;
use misc::*;
mod updates;
use updates::*;

use nannou::{prelude::*, wgpu::Device};
use nannou_egui::Egui;

const NUM_BOIDS: usize = 255;
const NUM_PREDATORS: usize = 4;

fn vertices_as_bytes(data: &[Vertex]) -> &[u8] {
    unsafe { wgpu::bytes::from_slice(data) }
}

fn main() {
    nannou::app(model).update(update).run();
}

fn model(app: &App) -> Model {
    let ui_wid = app
        .new_window()
        .title("ui-controls")
        .size(1366, 768)
        .view(ui_view)
        .raw_event(raw_ui_event)
        .build()
        .unwrap();

    let main_wid = app
        .new_window()
        .size(1366, 768)
        .view(view)
        //.raw_event(raw_ui_event)
        .build()
        .unwrap();

    let window = app.window(main_wid).unwrap();

    let ui_window = app.window(ui_wid).unwrap();
    let ui = Egui::from_window(&ui_window);

    let view_params = ViewParams {
        x_shift: 0.0,
        y_shift: 0.0,
        zoom: 0.01,
        time_modifier: 0.01,
    };

    let boid_params = BoidParams {
        max_velocity: 0.36400002,
        min_velocity: -0.35600003,
        turn_factor: 0.001,
        visual_range: 6.0,
        protected_range: 2.030001,
        centering_factor: 9.000001e-6,
        self_avoid_factor: 0.013000003,
        predator_avoid_factor: 0.01000003,
        matching_factor: 0.038800016,
    };

    let predator_params = PredatorParams {
        max_velocity: 4.6,
        min_velocity: -4.6,
        turn_factor: 0.001,
        pursuit_factor: 0.3,
        pursuit_multiplier: 1.2,
        matching_factor: 0.6,
        self_avoid_factor: 0.003000003,
        visual_range: 100.0,
        protected_range: 20.0,
        interest_range: 50.0,
    };

    let kcm = KeyboardControlMode::View;

    let device = window.device();
    let win_size = window.inner_size_points();
    let format = Frame::TEXTURE_FORMAT;
    let sample_count = window.msaa_samples();

    let vs_desc = wgpu::include_wgsl!("./shaders/vertex/v2.wgsl");
    let fs_desc = wgpu::include_wgsl!("./shaders/fragment/boids_frag.wgsl");
    let vs_mod = device.create_shader_module(vs_desc);
    let fs_mod = device.create_shader_module(fs_desc);

    let boid_pos_desc = wgpu::include_wgsl!("./shaders/compute/init_boids.wgsl");
    let boid_sac_desc = wgpu::include_wgsl!("./shaders/compute/seperation_alignment_cohesion.wgsl");
    let pred_pos_desc = wgpu::include_wgsl!("./shaders/compute/init_predator.wgsl");
    let pred_pursuit_desc = wgpu::include_wgsl!("./shaders/compute/predator_chase_path.wgsl");
    let boid_pos_mod = device.create_shader_module(boid_pos_desc);
    let boid_sac_mod = device.create_shader_module(boid_sac_desc);
    let pred_pos_mod = device.create_shader_module(pred_pos_desc);
    let pred_pursuit_mod = device.create_shader_module(pred_pursuit_desc);

    let vertices_bytes = vertices_as_bytes(&VERTICES[..]);
    let vertex_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("Vertex Buffer"),
        contents: vertices_bytes,
        usage: wgpu::BufferUsages::VERTEX,
    });

    let boids_pos_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Boids Positions Buffer"),
        size: (std::mem::size_of::<[Boid; NUM_BOIDS]>()) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let predator_pos_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Predator Positions Buffer"),
        size: (std::mem::size_of::<[Boid; NUM_PREDATORS]>()) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let pursuits_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("Pursuits IDs Buffer"),
        contents: unsafe { wgpu::bytes::from_slice(&[0xFFFFFFFFu32; NUM_PREDATORS]) },
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    });

    let captures_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("Predator Capture Count Buffer"),
        contents: unsafe { wgpu::bytes::from_slice(&[0u32; NUM_PREDATORS]) },
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    });

    let captured_boids_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("Captured Boid IDs Buffer"),
        contents: unsafe { wgpu::bytes::from_slice(&[1.0f32; NUM_BOIDS]) },
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    });

    let cpu_read_boids_pos_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("CPU Readable Buffer - Boids"),
        size: (std::mem::size_of::<[Boid; NUM_BOIDS]>()) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let cpu_read_predators_pos_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("CPU Readable Buffer - Predators"),
        size: (std::mem::size_of::<[Boid; NUM_PREDATORS]>()) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let cpu_read_predators_pursuits_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("CPU Readable Buffer - Predators Current Target Boid IDs"),
        size: (std::mem::size_of::<[u32; NUM_PREDATORS]>()) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let cpu_read_predators_captures_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("CPU Readable Buffer - Predators Current Target Boid IDs"),
        size: (std::mem::size_of::<[u32; NUM_PREDATORS]>()) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let layouts = get_bind_group_layouts(device);

    let render_pipeline =
        wgpu::RenderPipelineBuilder::from_layout(&layouts.render_pipeline_layout, &vs_mod)
            .fragment_shader(&fs_mod)
            .color_format(format)
            .add_vertex_buffer::<Vertex>(&wgpu::vertex_attr_array![0 => Float32x2])
            .sample_count(sample_count)
            .build(device);

    let compute_boid_pos_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Boid Position Pipeline"),
            layout: Some(&layouts.compute_pipeline_layout),
            module: &boid_pos_mod,
            entry_point: "compute_boid_positions",
        });

    let compute_predator_pos_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Predator Position Pipeline"),
            layout: Some(&layouts.compute_pipeline_layout),
            module: &pred_pos_mod,
            entry_point: "compute_predator_position",
        });

    let compute_sac_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute SAC Pipeline"),
        layout: Some(&layouts.compute_pipeline_layout),
        module: &boid_sac_mod,
        entry_point: "sac",
    });

    let compute_pursuit_curve_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Predator Pursuit Pipeline"),
            layout: Some(&layouts.compute_pipeline_layout),
            module: &pred_pursuit_mod,
            entry_point: "compute_predator_pursuit",
        });

    let time_uniform = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Time Uniform Buffer"),
        size: std::mem::size_of::<f32>() as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let resolution_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Resolution Uniform Buffer"),
        contents: bytemuck::cast_slice(&[win_size.0, win_size.1]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let (view_params_storage, boid_params_storage, predator_params_storage) =
        get_storage_buffers(&device, view_params, boid_params, predator_params);

    let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &layouts.uniform_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: time_uniform.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: resolution_uniform.as_entire_binding(),
            },
        ],
        label: Some("uniforms_bind_group"),
    });

    let variable_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &layouts.variable_bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: view_params_storage.as_entire_binding(),
        }],
        label: Some("params_bind_group"),
    });

    let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &layouts.compute_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: boids_pos_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: boid_params_storage.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: predator_pos_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: predator_params_storage.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: time_uniform.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: pursuits_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: captures_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: captured_boids_buf.as_entire_binding(),
            },
        ],
        label: Some("compute_bind_group"),
    });

    let queue = window.queue();

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Compute Encoder - Boids"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compute Pass - Boids"),
        });
        compute_pass.set_pipeline(&compute_boid_pos_pipeline);
        compute_pass.set_bind_group(0, &compute_bind_group, &[]);
        compute_pass.dispatch_workgroups(16, 16, 1);
    }

    encoder.copy_buffer_to_buffer(
        &boids_pos_buf,
        0,
        &cpu_read_boids_pos_buf,
        0,
        (std::mem::size_of::<[Boid; NUM_BOIDS]>()) as wgpu::BufferAddress,
    );

    queue.submit(Some(encoder.finish()));

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Compute Encoder - Predator"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compute Pass - Predator"),
        });
        compute_pass.set_pipeline(&compute_predator_pos_pipeline);
        compute_pass.set_bind_group(0, &compute_bind_group, &[]);
        compute_pass.dispatch_workgroups(4, 1, 1);
    }

    encoder.copy_buffer_to_buffer(
        &predator_pos_buf,
        0,
        &cpu_read_predators_pos_buf,
        0,
        (std::mem::size_of::<[Boid; NUM_PREDATORS]>()) as wgpu::BufferAddress,
    );

    queue.submit(Some(encoder.finish()));

    Model {
        pipelines: Pipelines {
            render_pipeline,
            compute_boid_pos_pipeline,
            compute_sac_pipeline,
            compute_predator_pos_pipeline,
            compute_pursuit_curve_pipeline,
        },
        compute_bind_group,
        buffers: Buffers {
            vertex_buf,
            boids_pos_buf,
            cpu_read_boids_pos_buf,
            predator_pos_buf,
            pursuits_buf,
            captures_buf,
            captured_boids_buf,
            cpu_read_predators_pos_buf,
            cpu_read_predators_pursuits_buf,
            cpu_read_predators_captures_buf,
        },
        uniforms: Uniforms {
            time_uniform,
            resolution_uniform,
            uniform_bind_group,
        },
        variables: Variables {
            view_params,
            view_params_storage,
            boid_params,
            predator_params,
            predator_params_storage,
            boid_params_storage,
            variable_bind_group,
        },
        controls: Controls { kcm },
        main_wid,
        ui,
    }
}

fn view(app: &App, model: &Model, frame: Frame) {
    let mut encoder = frame.command_encoder();
    let time = app.time as f32;
    let time_bytes = time.to_ne_bytes();
    let window = app.main_window();
    let queue = window.queue();

    if window.inner_size_points().0 != model.uniforms.resolution_uniform.size() as f32 {
        // Update the resolution uniform buffer with the new window size
        queue.write_buffer(
            &model.uniforms.resolution_uniform,
            0,
            bytemuck::cast_slice(&[window.inner_size_points().0, window.inner_size_points().1]),
        );
    }

    queue.write_buffer(&model.uniforms.time_uniform, 0, &time_bytes);

    let mut render_pass = wgpu::RenderPassBuilder::new()
        .color_attachment(frame.texture_view(), |color| color)
        .begin(&mut encoder);

    render_pass.set_bind_group(0, &model.compute_bind_group, &[]);
    render_pass.set_bind_group(1, &model.uniforms.uniform_bind_group, &[]);
    render_pass.set_bind_group(2, &model.variables.variable_bind_group, &[]);
    render_pass.set_pipeline(&model.pipelines.render_pipeline);
    render_pass.set_vertex_buffer(0, model.buffers.vertex_buf.slice(..));

    let vertex_range = 0..VERTICES.len() as u32;
    let instance_range = 0..1;
    render_pass.draw(vertex_range, instance_range);
}

fn raw_ui_event(_app: &App, model: &mut Model, event: &nannou::winit::event::WindowEvent) {
    model.ui.handle_raw_event(event);
}

fn ui_view(_app: &App, model: &Model, frame: Frame) {
    model.ui.draw_to_frame(&frame).unwrap();
}

fn get_bind_group_layouts(device: &Device) -> Layouts {
    let uniform_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<TimeUniform>() as _
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<ResolutionUniform>() as _,
                        ),
                    },
                    count: None,
                },
            ],
            label: Some("uniform_bind_group_layout"),
        });

    let variable_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<ViewParams>() as _),
                },
                count: None,
            }],
            label: Some("variable_bind_group_layout"),
        });

    let compute_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<Boid>() as _),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<BoidParams>() as _
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<Boid>() as _),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<PredatorParams>() as _,
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<TimeUniform>() as _
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<
                            &[u32; NUM_PREDATORS],
                        >() as _),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<
                            &[u32; NUM_PREDATORS],
                        >() as _),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<&[f32; NUM_BOIDS]>() as _,
                        ),
                    },
                    count: None,
                },
            ],
            label: Some("compute_bind_group_layout"),
        });

    let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Unified Pipeline Layout"),
        bind_group_layouts: &[
            &compute_bind_group_layout,
            &uniform_bind_group_layout,
            &variable_bind_group_layout,
        ],
        push_constant_ranges: &[],
    });

    let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Compute Pipeline Layout"),
        bind_group_layouts: &[&compute_bind_group_layout],
        push_constant_ranges: &[],
    });

    Layouts {
        uniform_bind_group_layout,
        variable_bind_group_layout,
        compute_bind_group_layout,
        render_pipeline_layout,
        compute_pipeline_layout,
    }
}

fn get_storage_buffers(
    device: &Device,
    view_params: ViewParams,
    boid_params: BoidParams,
    predator_params: PredatorParams,
) -> (wgpu::Buffer, wgpu::Buffer, wgpu::Buffer) {
    let view_params_storage = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Parameters Storage Buffer"),
        contents: bytemuck::cast_slice(&[
            view_params.x_shift,
            view_params.y_shift,
            view_params.zoom,
            view_params.time_modifier,
        ]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let boid_params_storage = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Boid Parameters Storage Buffer"),
        contents: bytemuck::cast_slice(&[
            boid_params.max_velocity,
            boid_params.min_velocity,
            boid_params.turn_factor,
            boid_params.visual_range,
            boid_params.protected_range,
            boid_params.centering_factor,
            boid_params.self_avoid_factor,
            boid_params.predator_avoid_factor,
            boid_params.matching_factor,
        ]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let predator_params_storage = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Boid Parameters Storage Buffer"),
        contents: bytemuck::cast_slice(&[
            predator_params.max_velocity,
            predator_params.min_velocity,
            predator_params.turn_factor,
            predator_params.pursuit_factor,
            predator_params.pursuit_multiplier,
            predator_params.matching_factor,
            predator_params.self_avoid_factor,
            predator_params.visual_range,
            predator_params.protected_range,
            predator_params.interest_range,
        ]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    (
        view_params_storage,
        boid_params_storage,
        predator_params_storage,
    )
}
