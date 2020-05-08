use rand;
use std::f32::consts::E;
use std::path::Path;

use sdl2::rect::{Point, Rect};
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::pixels::Color;
use sdl2::video::{Window, WindowContext};
use sdl2::render::Canvas;
use sdl2::render::TextureCreator;
use sdl2::ttf::Font;

#[derive(Debug)]
struct Layer {
    pub activations : Vec<f32>,
    pub weighted_sums : Vec<f32>,
    pub sigmoid_derivatives : Vec<f32>,
    pub biases : Vec<f32>,
    pub errors : Vec<f32>,
    pub weights : Vec<Vec<f32>>,
    pub weights_gradient : Vec<Vec<f32>>
}
impl Layer {
    pub fn new() -> Layer{
        Layer {
            activations : Vec::new(),
            weighted_sums : Vec::new(),
            sigmoid_derivatives : Vec::new(),
            biases : Vec::new(),
            errors : Vec::new(),
            weights : Vec::new(),
            weights_gradient : Vec::new(),
        }
    }
}

#[derive(Debug)]
struct NeuralNet {
    pub layers : Vec<Layer>,
    pub learn_rate : f32,
}

impl NeuralNet {
    pub fn new(topology : Vec<u32>, learn_rate : f32) -> NeuralNet {
        let mut nn = NeuralNet {
            layers : Vec::new(),
            learn_rate : learn_rate,
        };
        for i in 0..topology.len() {
            nn.layers.push(Layer::new());
            for n in 0..topology[i] {
                nn.layers[i].activations.push(0.0);
                nn.layers[i].biases.push(rand::random::<f32>());
                nn.layers[i].weighted_sums.push(0.0);
                nn.layers[i].errors.push(0.0);
                nn.layers[i].sigmoid_derivatives.push(0.0);
                //if not the input layer
                if i != 0 {
                    nn.layers[i].weights.push(Vec::new());
                    nn.layers[i].weights_gradient.push(Vec::new());
                    for x in 0..topology[i - 1] {
                        nn.layers[i].weights[n as usize].push(rand::random::<f32>());
                        nn.layers[i].weights_gradient[n as usize].push(0.0);
                    }
                }
            }
        }
        nn
    }

    pub fn feed_forward(&mut self, input : &Vec<f32>) {
        //check if no. of neuron in input layer is same as no. of inputs
        if self.layers[0].activations.len() == input.len() {
            //giving input to input layer
            for i in 0..self.layers[0].activations.len() {
                self.layers[0].activations[i] = input[i];
            }
            for l in 1..self.layers.len() {
                for n in 0..self.layers[l].activations.len() {
                    let weighted_sum = weighted_sum(&self.layers[l].weights[n], &self.layers[l - 1].activations) + self.layers[l].biases[n];
                    self.layers[l].activations[n] = sigmoid(weighted_sum);
                    self.layers[l].sigmoid_derivatives[n] = sigmoid_derivative(weighted_sum);
                    self.layers[l].weighted_sums[n] = weighted_sum;
                }
            }
        }
    }

    pub fn back_propagate(&mut self, target : &Vec<f32>) {

        let output_layer_index = self.layers.len() - 1;

        //checking for dimension of output layer and target
        if self.layers[output_layer_index].activations.len() == target.len() {
            //computing error in the output layer
            for i in 0..self.layers[output_layer_index].activations.len() {
                self.layers[output_layer_index].errors[i] = 2.0 * (self.layers[output_layer_index].activations[i] - target[i])
                                                                        * self.layers[output_layer_index].sigmoid_derivatives[i];
                println!("output : {:?}, cost : {:?}",  self.layers[output_layer_index].activations[i],
                                                            (self.layers[output_layer_index].activations[i] - target[i]).powf(2.0));
            }
            for l in (self.layers.len() - 2)..0 {
                self.layers[l].errors = hadamard_product(&multiply_vec(&transpose(&self.layers[l+1].weights), &self.layers[l+1].errors),
                                            &self.layers[l].sigmoid_derivatives);
            }

            for l in 0..self.layers.len() {
                if l != 0 {
                    for i in 0..self.layers[l].weights.len() {
                        for j in 0..self.layers[l].weights[i].len() {
                            self.layers[l].weights_gradient[i][j] = self.layers[l].errors[i] * self.layers[l-1].activations[j];
                            self.layers[l].weights[i][j] -= self.learn_rate * self.layers[l].weights_gradient[i][j];
                        }
                        self.layers[l].biases[i] -= self.learn_rate * self.layers[l].errors[i];
                    }
                }
            }

        }
    }

    pub fn log(&self) {
        for l in 0..self.layers.len() {
            println!("layer : {:?}", self.layers[self.layers.len() - 1]);
        }
    }
}

fn sigmoid(z : f32) -> f32 {
    return 1.0 / (1.0 + E.powf(-z));
}

fn sigmoid_derivative(z : f32) -> f32 {
    return sigmoid(z) / (1.0 - sigmoid(z));
}

fn weighted_sum(w : &Vec<f32>,  a : &Vec<f32>) -> f32{
    let mut sum = 0.0;
    for i in 0..w.len() {
        sum += w[i] * a[i];
    }
    return sum;
}

fn multiply_vec(w : &Vec<Vec<f32>>, v : &Vec<f32>) -> Vec<f32>{
    let mut result = Vec::new();
    let mut sum = 0.0;
    for i in 0..v.len() {
        sum = 0.0;
        for j in 0..v.len() {
            sum += w[i][j] * v[j];
        }
        result.push(sum);
    }
    return result;
}

fn transpose(v : &Vec<Vec<f32>>) -> Vec<Vec<f32>>{
    let mut transposed = Vec::new();
    for i in 0..v[0].len() {
        transposed.push(Vec::new());
        for j in 0..v.len() {
            transposed[i].push(v[j][i]);
        }
    }
    return transposed;
}

fn hadamard_product(w : &Vec<f32>, v : &Vec<f32>) -> Vec<f32>{
    let mut product = Vec::new();
    for i in 0..w.len() {
        product.push(w[i] * v[i]);
    }

    return product;
}

fn render_neural_net(nn : &NeuralNet, canvas : &mut Canvas<Window>, texture_creator : &mut TextureCreator<WindowContext>, font : &mut Font) {

    let origin = (100, 100);
    let mut x = origin.0;
    let mut y = origin.1;
    let mut v = 0;

    let size : i32 = 40;
    let offset : i32 = 80;

    for l in 0..nn.layers.len() {
        y = origin.0;
        for n in 0..nn.layers[l].activations.len() {
            canvas.set_draw_color(Color::RGB(255, 255, 0));
            if l < nn.layers.len()-1 {
                v = origin.0;
                for _i in 0..nn.layers[l+1].activations.len() {
                    let _x = x + size / 2;
                    let _y = y + size / 2;
                    let _v = v + size / 2;
                    canvas.draw_line(Point::new(_x, _y), Point::new(_x + size + offset, _v)).unwrap();
                    v += size + offset;
                }
            }
            let surface = font.render(&format!("{:.3}", nn.layers[l].activations[n])).blended(Color::RGBA(255, 255, 255, 255)).map_err(|e| e.to_string()).unwrap();
            let font_texture = texture_creator.create_texture_from_surface(&surface).map_err(|e| e.to_string()).unwrap();

            let width = font_texture.query().width;
            let height = font_texture.query().height;

            canvas.set_draw_color(Color::RGB(0, 0, 0));
            canvas.fill_rect(Rect::new(x, y, size as u32, size as u32)).unwrap();
            canvas.set_draw_color(Color::RGB(255, 0, 0));
            canvas.draw_rect(Rect::new(x, y, size as u32, size as u32)).unwrap();
            canvas.copy(&font_texture, None, Some(Rect::new(x + 5, y + 10, width, height)));
            y += size + offset;
        }
        x += size + offset;
    }
}


fn main() {
    let input = vec![vec![0.1, 0.3],
                    vec![0.2, 0.4],
                    vec![0.5, 0.7],
                    vec![0.6, 0.8]];

    let target = vec![vec![0.1], vec![0.2], vec![0.3], vec![0.4]];
    //
    let topology = vec![2, 4, 5, 5, 5, 4, 2];
    let mut nn = NeuralNet::new(topology, 0.5);
    //
    // let mut index = 0;
    // for i in 0..1000 {
        //
        //     println!("{:?}, {:?}", &input[index], &target[index]);
     nn.feed_forward(&input[0]);
        //     // nn.log();
        //     // println!();
        //     nn.back_propagate(&target[index]);
        //     //nn.log();
        //
        //     index += 1;
        //     if index == 4 {
            //         index = 0;
            //     }
            // }

    let sdl_context = sdl2::init().unwrap();
    let ttf_context = sdl2::ttf::init().map_err(|e| e.to_string()).unwrap();
    let video_subsystem = sdl_context.video().unwrap();
    let window = video_subsystem.window("Neural Net visualization", 1280, 720)
      .position_centered()
      .build()
      .map_err(|e| e.to_string()).unwrap();
      let mut running = true;

    let mut canvas = window.into_canvas().software().build().map_err(|e| e.to_string()).unwrap();
    let mut event_pump = sdl_context.event_pump().unwrap();
    let mut texture_creator = canvas.texture_creator();

    let mut font = ttf_context.load_font(Path::new("font/OpenSans-Regular.ttf"), 12).unwrap();

    while running {
        for event in event_pump.poll_iter() {
	        match event {
	            Event::Quit {..} | Event::KeyDown {keycode: Some(Keycode::Escape), ..} => {
	                    running = false;
	                },
	            _=>(),
	        }
    	}
        canvas.set_draw_color(Color::RGB(0, 0, 0));
        canvas.clear();
        render_neural_net(&nn, &mut canvas, &mut texture_creator, &mut font);
        canvas.present();
        //break;
    }
}
