use crate::{
    update_boid_params_buffer, update_boid_props, update_predator_params_buffer,
    update_view_params_buffer, Boid, KeyboardControlMode, Model,
};
use bytemuck::Pod;
use futures::executor::block_on;
use nannou::event::Key;
use nannou::wgpu::Buffer;
use nannou::{wgpu, App};
use std::thread;
use std::time::Duration;

pub fn check_keys(app: &App, model: &mut Model) {
    let pressed = &app.keys.down;

    if pressed.contains(&Key::Key1) {
        model.controls.kcm = KeyboardControlMode::Boids;
        println!("kcm: {:?}", model.controls.kcm);
    } else if pressed.contains(&Key::Key2) {
        model.controls.kcm = KeyboardControlMode::Predator;
        println!("kcm: {:?}", model.controls.kcm);
    } else if pressed.contains(&Key::Key3) {
        model.controls.kcm = KeyboardControlMode::Debug;
        println!("kcm: {:?}", model.controls.kcm);
    } else if pressed.contains(&Key::Key4) {
        model.controls.kcm = KeyboardControlMode::View;
        println!("kcm: {:?}", model.controls.kcm);
    }

    match model.controls.kcm {
        KeyboardControlMode::View => view_controls(app, model),
        KeyboardControlMode::Boids => boids_controls(app, model),
        KeyboardControlMode::Predator => predator_controls(app, model),
        KeyboardControlMode::Debug => debug_controls(app, model),
    }
}

fn boids_controls(app: &App, model: &mut Model) {
    let pressed = &app.keys.down;
    let mut dval = 0.0f32;

    if pressed.contains(&Key::Up) {
        dval = 1.0f32;
    }

    if pressed.contains(&Key::Down) {
        dval = -1.0f32;
    }

    if pressed.contains(&Key::Plus) {
        let maxv = &mut model.variables.boid_params.max_velocity;
        *maxv = f32::max(0.1, *maxv + (0.003 * dval));
        update_boid_params_buffer(app, model);
    }
    if pressed.contains(&Key::Minus) {
        let minv = &mut model.variables.boid_params.min_velocity;
        *minv = f32::max(0.0, *minv + (0.003 * dval));
        update_boid_params_buffer(app, model);
    }
    if pressed.contains(&Key::T) {
        let tf = &mut model.variables.boid_params.turn_factor;
        *tf = f32::max(0.0, *tf + (0.003 * dval));
        update_boid_params_buffer(app, model);
    } else if pressed.contains(&Key::V) {
        let vr = &mut model.variables.boid_params.visual_range;
        *vr = f32::max(0.0, *vr + (0.01 * dval));
        update_boid_params_buffer(app, model);
    } else if pressed.contains(&Key::P) {
        let pr = &mut model.variables.boid_params.protected_range;
        *pr = f32::max(0.0, *pr + (0.01 * dval));
        update_boid_params_buffer(app, model);
    } else if pressed.contains(&Key::C) {
        let cf = &mut model.variables.boid_params.centering_factor;
        *cf = f32::max(0.0, *cf + (0.0000003 * dval));
        update_boid_params_buffer(app, model);
    } else if pressed.contains(&Key::A) {
        let af = &mut model.variables.boid_params.self_avoid_factor;
        *af = f32::max(0.0, *af + (0.0003 * dval));
        update_boid_params_buffer(app, model);
    } else if pressed.contains(&Key::P) {
        let af = &mut model.variables.boid_params.predator_avoid_factor;
        *af = f32::max(0.0, *af + (0.0003 * dval));
        update_boid_params_buffer(app, model);
    } else if pressed.contains(&Key::M) {
        let mf = &mut model.variables.boid_params.matching_factor;
        *mf = f32::max(0.0, *mf + (0.0003 * dval));
        update_boid_params_buffer(app, model);
    }
}

fn predator_controls(app: &App, model: &mut Model) {
    let pressed = &app.keys.down;

    let mut dval = 0.0f32;

    if pressed.contains(&Key::Up) {
        dval = 1.0f32;
    }

    if pressed.contains(&Key::Down) {
        dval = -1.0f32;
    }

    if pressed.contains(&Key::Plus) {
        let maxv = &mut model.variables.predator_params.max_velocity;
        *maxv = f32::max(0.1, *maxv + (0.03 * dval));
        update_predator_params_buffer(app, model);
    } else if pressed.contains(&Key::Minus) {
        let minv = &mut model.variables.predator_params.min_velocity;
        *minv = f32::max(0.0, *minv + (0.03 * dval));
        update_predator_params_buffer(app, model);
    } else if pressed.contains(&Key::T) {
        let tf = &mut model.variables.predator_params.turn_factor;
        *tf = f32::max(0.0, *tf + (0.003 * dval));
        update_predator_params_buffer(app, model);
    } else if pressed.contains(&Key::V) {
        let vr = &mut model.variables.predator_params.visual_range;
        *vr = f32::max(0.0, *vr + (0.001 * dval));
        update_predator_params_buffer(app, model);
    } else if pressed.contains(&Key::I) {
        let vr = &mut model.variables.predator_params.interest_range;
        *vr = f32::max(0.0, *vr + (0.001 * dval));
        update_predator_params_buffer(app, model);
    } else if pressed.contains(&Key::P) {
        let pf = &mut model.variables.predator_params.pursuit_factor;
        *pf = f32::max(0.0, *pf + (0.001 * dval));
        update_predator_params_buffer(app, model);
    } else if pressed.contains(&Key::O) {
        let pf = &mut model.variables.predator_params.pursuit_multiplier;
        *pf = f32::max(0.0, *pf + (0.05 * dval));
        update_predator_params_buffer(app, model);
    } else if pressed.contains(&Key::M) {
        let mf = &mut model.variables.predator_params.matching_factor;
        *mf = f32::max(0.0, *mf + (0.001 * dval));
        update_predator_params_buffer(app, model);
    }
}

fn debug_controls(app: &App, model: &mut Model) {
    // PRINT CURRENT FRAME --------------------------------------------------------
    if app.keys.down.contains(&Key::Space) {
        let file_path = app
            .project_path()
            .expect("failed to locate project directory")
            .join("frames")
            .join(format!("{:0}.png", app.elapsed_frames()));
        app.main_window().capture_frame(file_path);
    }

    // PRINT CURRENT PARAMETER VALUES ----------------------------------------------
    if app.keys.down.contains(&Key::I) {
        println!("\nview_params:\n{:#?}\n", model.variables.view_params);
        thread::sleep(Duration::from_millis(50));
    } else if app.keys.down.contains(&Key::B) {
        println!("\nboids_params:\n{:#?}", model.variables.boid_params);
        thread::sleep(Duration::from_millis(50));
    } else if app.keys.down.contains(&Key::V) {
        println!("\npredator_params:\n{:#?}", model.variables.predator_params);
    } else if app.keys.down.contains(&Key::Comma) {
        print_gpu_data::<Boid>(app, &model.buffers.cpu_read_boids_pos_buf, "Boid");
        thread::sleep(Duration::from_millis(50));
    } else if app.keys.down.contains(&Key::Semicolon) {
        print_gpu_data::<Boid>(app, &model.buffers.cpu_read_predators_pos_buf, "Predator");
        thread::sleep(Duration::from_millis(50));
    } else if app.keys.down.contains(&Key::P) {
        print_gpu_data::<[u32; 4]>(
            app,
            &model.buffers.cpu_read_predators_pursuits_buf,
            "Pursuit IDs",
        );
        thread::sleep(Duration::from_millis(50));
    } else if app.keys.down.contains(&Key::C) {
        print_gpu_data::<[f32; 4]>(
            app,
            &model.buffers.cpu_read_predators_captures_buf,
            "Captures per Predator",
        );
        thread::sleep(Duration::from_millis(50))
    }
}
fn view_controls(app: &App, model: &mut Model) {
    if app.keys.down.contains(&Key::Left) {
        model.variables.view_params.x_shift -= 0.01 / model.variables.view_params.zoom;
        update_view_params_buffer(app, model);
    } else if app.keys.down.contains(&Key::Right) {
        model.variables.view_params.x_shift += 0.01 / model.variables.view_params.zoom;
        update_view_params_buffer(app, model);
    } else if app.keys.down.contains(&Key::Up) {
        model.variables.view_params.y_shift += 0.01 / model.variables.view_params.zoom;
        update_view_params_buffer(app, model);
    } else if app.keys.down.contains(&Key::Down) {
        model.variables.view_params.y_shift -= 0.01 / model.variables.view_params.zoom;
        update_view_params_buffer(app, model);
    } else if app.keys.down.contains(&Key::X) {
        let mz = model.variables.view_params.zoom;
        model.variables.view_params.zoom -= 0.1 * mz;
        update_view_params_buffer(app, model);
    } else if app.keys.down.contains(&Key::Y) {
        let mz = model.variables.view_params.zoom;
        model.variables.view_params.zoom += 0.1 * mz;
        update_view_params_buffer(app, model);
    }
}

pub(crate) fn print_gpu_data<T: Pod + std::fmt::Debug>(
    app: &App,
    buffer: &Buffer,
    obj_label: &str,
) {
    let mw = app.main_window();
    let device = mw.device();

    // Map the buffer for reading
    let buffer_slice = buffer.slice(..);
    let (tx, rx) = futures::channel::oneshot::channel();

    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).unwrap();
    });

    // Wait for the GPU to finish executing the commands
    device.poll(wgpu::Maintain::Wait);
    // Wait for the buffer to be mapped
    let result = block_on(rx);

    match result {
        Ok(_) => {
            let buf_view = buffer_slice.get_mapped_range();
            let data: &[T] = bytemuck::cast_slice(&buf_view);

            // Print the boids current properties
            for (i, obj) in data.iter().enumerate() {
                println!("{} {}:\n{:?}", obj_label, i, obj);
            }

            drop(buf_view);
            buffer.unmap();
        }
        Err(e) => eprintln!("Error retrieving gpu data: {:?}", e),
    }
}
