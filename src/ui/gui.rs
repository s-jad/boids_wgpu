use std::error::Error;

use bytemuck::Pod;
use futures::executor::block_on;
use nannou::{
    event::{Key, Update},
    wgpu::{self, Buffer, BufferView},
    App,
};
use nannou_egui::egui::{self, epaint::Shadow};

use crate::{update_view_params_buffer, Model, NUM_PREDATORS};

use super::check_keys;

pub fn update_ui<T>(app: &App, m: &mut Model, u: Update)
where
    T: Pod + std::fmt::Debug,
{
    let mut reset = false;
    let predator_stats: Result<Vec<u32>, Box<dyn Error>>;

    {
        predator_stats = get_predator_data(app, &m.buffers.cpu_read_predators_captures_buf);
    }

    let res = match predator_stats {
        Ok(ps) => ps,
        Err(error) => {
            println!("Error fetching predator stats for gui: {error}");
            vec![0u32; NUM_PREDATORS]
        }
    };

    {
        let ui = &mut m.ui;
        ui.set_elapsed_time(u.since_start);
        let ctx = ui.begin_frame();

        egui::Window::new("Controls")
            .frame(egui::Frame {
                fill: egui::Color32::from_rgb(24, 20, 23),
                inner_margin: egui::Vec2::new(20.0, 10.0).into(),
                rounding: 10.0.into(),
                ..Default::default()
            })
            .movable(false)
            .show(&ctx, |ui| {
                ui.style_mut().spacing.item_spacing = egui::Vec2::new(20.0, 20.0);
                ui.style_mut().spacing.button_padding = egui::Vec2::new(15.0, 5.0);
                ui.style_mut().spacing.window_margin = egui::Vec2::new(15.0, 10.0).into();
                ui.style_mut().visuals.widgets.inactive.bg_fill =
                    egui::Color32::from_rgb(244, 200, 230);

                egui::Frame::dark_canvas(ui.style())
                    .inner_margin(egui::Vec2::new(15.0, 10.0))
                    .shadow(Shadow::small_dark())
                    .rounding(10.0)
                    .show(ui, |ui| {
                        ui.colored_label(
                            nannou_egui::egui::Rgba::from_rgb(1.0, 0.4, 0.34),
                            "Capture Scores:",
                        );

                        egui::Grid::new("view_params")
                            .spacing(egui::Vec2::new(20.0, 10.0))
                            .show(ui, |ui| {
                                ui.label("Predator 0:");
                                ui.label(format!("{}", res[0]));
                                ui.label("Predator 1:");
                                ui.label(format!("{}", res[1]));
                                ui.label("Predator 2:");
                                ui.label(format!("{}", res[2]));
                                ui.label("Predator 3:");
                                ui.label(format!("{}", res[3]));
                            });
                    });

                if ui.button("run").clicked() {
                    reset = true;
                }
            });
    }

    {
        check_keys(app, m);
    }
}

pub(crate) fn get_predator_data<'a, T: Pod + std::fmt::Debug>(
    app: &'a App,
    buffer: &'a Buffer,
) -> Result<Vec<T>, Box<dyn std::error::Error>> {
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

            let owned_data = data.to_owned();
            drop(buf_view);
            buffer.unmap();

            Ok(owned_data)
        }
        Err(e) => {
            eprintln!("Error retrieving gpu data: {:?}", e);
            Err(Box::new(e))
        }
    }
}
