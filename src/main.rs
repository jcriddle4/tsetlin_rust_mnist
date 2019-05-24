

#![allow(unused_parens)]
#![allow(non_snake_case)]

mod tsetlin;

use std::mem;
use byteorder::{LittleEndian};
use byteorder::ByteOrder;

use std::env;
use std::io;
use std::io::prelude::*;
use std::io::BufReader;
use std::fs::File;
use std::time::Instant;
use std::process;
use std::thread;
use std::cmp::Ordering;

extern crate rand;
extern crate rayon;

use rand::prelude::*;
use rayon::prelude::*;

use tsetlin::*;
//use std::sync::mpsc::channel;
use core::borrow::Borrow;


const EPOCHS: u64 = 400;
const SVALUE: u64 = 11;

const NUMBER_OF_TRAINING_EXAMPLES: usize = 60000;
const NUMBER_OF_TEST_EXAMPLES: usize = 10000;

fn read_file(
    file_name: &str
    , num_examples: usize
    , x_vec: &mut Vec<Vec<u64>>
    , y_vec: &mut Vec<u64>
    , la_chunks: usize
    , features: u64
    )
{
    assert!(x_vec.len() == 0 && y_vec.len() == 0);

    println!("starting initilization of vectors");

    while x_vec.len() < num_examples{
        x_vec.push(vec![0; la_chunks]);
        y_vec.push(0);
    }

    let f = match File::open(file_name) {
        Err(_) => match File::open( format!("{}{}","../",file_name)) {  //We didn't find the file so we are looking up one directory higher to see if we can find the file.
            Err(_) => {println!("Could not open training data file:{}   Has the zip file been unzipped?",file_name); process::exit(0)},
            Ok(t) => t,
        }
        Ok(t) => t,
    };

    let mut reader = BufReader::new(f);
    let mut buffer = String::new();

    for i in 0..num_examples{
        buffer.clear();
        match reader.read_line(&mut buffer){
            Err(_) => {println!("Error reading line from training data file:{}",file_name); return}
            Ok(_) => ()
        }

        let line: Vec<&str> = buffer.split(" ").collect();

        for j in 0..features+1{
            let line_ptr = j as usize;
            match line[line_ptr].trim().parse::<u64>(){
                Ok(n) => {
                    if j == features { //The actual digit is the last number on the line.
                        y_vec[i] = n;
                    } else if n == 1 {
                        let chunk_nr = (j / INT_SIZE) as usize;
                        let chunk_pos = (j % INT_SIZE) as usize;
                        x_vec[i][chunk_nr] |= (1 << chunk_pos);
                    } else {
                        let chunk_nr = ((j + features) / INT_SIZE) as usize;
                        let chunk_pos = ((j + features) % INT_SIZE) as usize;
                        x_vec[i][chunk_nr] |= (1 << chunk_pos);
                    }
                }
                Err(_) => {
                    println!("Error parsing file for numbers"); return
                }
            }
        }
    }
}

fn output_digit(x_vec: & Vec<Vec<u64>>, y_vec: & Vec<u64>, nth: usize){
    //let la_chunk: Vec<u64> = x_vec[nth];
    let y = y_vec[nth];
    for y in 0..28{
        for x in 0..28{
            let chunk_nr = ((x + y*28) / INT_SIZE) as usize;
            let chunk_pos = ((x + y*28) % INT_SIZE) as usize;

            if((x_vec[nth])[chunk_nr] & (1 << chunk_pos) > 0) {
                print!("@");
            } else {
                print!(".");
            }
        }
        println!("");
    }
    println!(" digit is supposed to be:{} at position:{}",y,nth)
}


fn shuffle_vecs(x_train: &mut Vec<Vec<u64>>, y_train: &mut Vec<u64>, rng: &mut ThreadRng){
    for _ in 0..2 {
        for i in 0..x_train.len() - 1 {
            let mut r: usize = rng.gen();
            r = r % x_train.len();

            if (r != i) {
                x_train.swap(i, r);
                y_train.swap(i, r);
            }
        }
    }
}

fn digit_tests(rng: &mut ThreadRng){
    let mut tm = init_tm(50,784,2000,10,8, 1234567, 10.0);

    let mut x_train: Vec<Vec<u64>> = Vec::with_capacity(NUMBER_OF_TRAINING_EXAMPLES);
    let mut y_train: Vec<u64> = Vec::with_capacity(NUMBER_OF_TRAINING_EXAMPLES);

    let mut x_train2: Vec<Vec<u64>> = Vec::with_capacity(NUMBER_OF_TEST_EXAMPLES);;
    let mut y_train2: Vec<u64> = Vec::with_capacity(NUMBER_OF_TEST_EXAMPLES);

    let mut x_test: Vec<Vec<u64>> = Vec::with_capacity(NUMBER_OF_TEST_EXAMPLES);;
    let mut y_test: Vec<u64> = Vec::with_capacity(NUMBER_OF_TEST_EXAMPLES);

    test_d3(&mut tm.tclasses[0]);

    read_file("/home/jcriddle/fast-tsetlin-machine-with-mnist-demo/MNISTTraining.txt"
              ,NUMBER_OF_TRAINING_EXAMPLES
              , &mut x_train
              , &mut y_train
              , tm.la_chunks
              , tm.features
    );

    read_file("/home/jcriddle/fast-tsetlin-machine-with-mnist-demo/MNISTTrainingSampled.txt"
              ,NUMBER_OF_TEST_EXAMPLES
              , &mut x_train2
              , &mut y_train2
              , tm.la_chunks
              , tm.features
    );

    read_file("/home/jcriddle/fast-tsetlin-machine-with-mnist-demo/MNISTTest.txt"
              ,NUMBER_OF_TEST_EXAMPLES
              , &mut x_test
              , &mut y_test
              , tm.la_chunks
              , tm.features
    );

    let r:usize = rng.gen();
    output_digit(& x_train, & y_train,r % NUMBER_OF_TRAINING_EXAMPLES);
    //output_digit(& x_train2, & y_train2,r % NUMBER_OF_TEST_EXAMPLES);
    //output_digit(& x_test, & y_test,r % NUMBER_OF_TEST_EXAMPLES);

    for i in 0..EPOCHS{
        let cwd = match env::current_dir(){
            Err(_) => {println!("Failed to find current working directory"); return}
            Ok(p) => p
        };
        println!(" epoch:{}  current directory {}",i,cwd.display());

        let now = Instant::now();

        // NUMBER_OF_TRAINING_EXAMPLES
        println!("Doing fit operation");
        shuffle_vecs(&mut x_train, &mut y_train, rng);
        tm_fit(&mut tm,&x_train, &y_train, NUMBER_OF_TRAINING_EXAMPLES,  rng);

        println!("Training accuracy");
        let train_accuracy = tm_eval(&mut tm, &mut x_train2, & y_train2, NUMBER_OF_TEST_EXAMPLES,1);
        let test_accuracy = tm_eval(&mut tm, &mut x_test, & y_test, NUMBER_OF_TEST_EXAMPLES,2);
        println!(" training results:{:.3}",train_accuracy * 100.0);
        println!(" test results:{:.3}",test_accuracy * 100.0);

        let elapsed = now.elapsed();
        let sec = (elapsed.as_secs() as f64) + (elapsed.subsec_nanos() as f64 / 1000_000_000.0);
        println!("Seconds: {}", sec);

        println!("");
    }
}


fn do_threads(tm: &mut TsetlinMachine, x_train: & Vec<Vec<u64>>, y_train: & Vec<u64>, num_examples: usize){
    let mut rng = rand::thread_rng();

    tm_fit(tm, & x_train, &y_train, num_examples, &mut rng);
    //println!("finishing thread id_num:{}",tm.id_num);
}

fn multi_threads(){
    let mut rng = rand::thread_rng();

    let mut tm_array: Vec<TsetlinMachine>  = Vec::with_capacity(8);

    for i in 0..8 {
        tm_array.push(init_tm(50, 784, 2000, 10, 8, i + 1, 10.0 + ((i as f64) / 2.0)));
    }

    let mut x_train: Vec<Vec<u64>> = Vec::with_capacity(NUMBER_OF_TRAINING_EXAMPLES);
    let mut y_train: Vec<u64> = Vec::with_capacity(NUMBER_OF_TRAINING_EXAMPLES);

    let mut x_train2: Vec<Vec<u64>> = Vec::with_capacity(NUMBER_OF_TEST_EXAMPLES);;
    let mut y_train2: Vec<u64> = Vec::with_capacity(NUMBER_OF_TEST_EXAMPLES);

    let mut x_test: Vec<Vec<u64>> = Vec::with_capacity(NUMBER_OF_TEST_EXAMPLES);;
    let mut y_test: Vec<u64> = Vec::with_capacity(NUMBER_OF_TEST_EXAMPLES);

    read_file("MNISTTraining.txt"
          ,NUMBER_OF_TRAINING_EXAMPLES
          , &mut x_train
          , &mut y_train
          , tm_array[0].la_chunks
          , tm_array[0].features
    );

    read_file("MNISTTrainingSampled.txt"
          ,NUMBER_OF_TEST_EXAMPLES
          , &mut x_train2
          , &mut y_train2
          , tm_array[0].la_chunks
          , tm_array[0].features
    );

    read_file("MNISTTest.txt"
          ,NUMBER_OF_TEST_EXAMPLES
          , &mut x_test
          , &mut y_test
          , tm_array[0].la_chunks
          , tm_array[0].features
    );

    for loop_count in 0..EPOCHS {
        let now = Instant::now();

        let cwd = match env::current_dir(){
            Err(_) => {println!("Failed to find current working directory"); return}
            Ok(p) => p
        };
        println!(" epoch:{}  current directory {} ",loop_count,cwd.display());

        tm_array.par_iter_mut().for_each(|tm| { do_threads(tm, &x_train, &y_train, x_train.len()) });

        shuffle_vecs(&mut x_train, &mut y_train, &mut rng);

        for i in 0..tm_array.len(){
            tm_array[i].life += 1;
        }

        println!("starting y_test eval");
        let score_idx = 1;
        tm_array.par_iter_mut().for_each(|tm| { let _ = tm_eval(tm, &x_test, &y_test, y_test.len(), score_idx); });
        //tm_array.sort_by(|b, a| a.score[score_idx].partial_cmp(&b.score[score_idx]).unwrap());

        println!("starting y_train2 eval");
        let score_idx = 2;
        tm_array.par_iter_mut().for_each(|tm| { let _ = tm_eval(tm, &x_train2, &y_train2, y_train2.len(), score_idx); });
        //tm_array.sort_by(|b, a| a.score[score_idx].partial_cmp(&b.score[score_idx]).unwrap());

        println!("starting y_train eval");
        let score_idx = 0;
        tm_array.par_iter_mut().for_each(|tm| { let _ = tm_eval(tm, &x_train, &y_train, y_train.len(), score_idx); });
        tm_array.sort_by(|b, a| a.score[score_idx].partial_cmp(&b.score[score_idx]).unwrap());
        for i in 0..tm_array.len() {
            println!(" Scores id_num:{}  life:{}  score0:{:.3}  score1:{:.3}  score2:{:.3}"
                     , tm_array[i].id_num
                     , tm_array[i].life
                     , tm_array[i].score[0] * 100.0
                     , tm_array[i].score[1] * 100.0
                     , tm_array[i].score[2] * 100.0
            );
        }

        //NOTE: We must have sorting on just the initial training set otherwise we are using "data" the machine shouldn't have access to.
        let mut tkill_a = tm_array.pop().unwrap();
        let mut tkill_b = tm_array.pop().unwrap();
        let mut tkill_c = tm_array.pop().unwrap();

        breed(&tm_array[0], &tm_array[1], &mut tkill_a, &mut rng);
        breed(&tm_array[1], &tm_array[2], &mut tkill_b, &mut rng);
        breed(&tm_array[2], &tm_array[3], &mut tkill_c, &mut rng);

        tm_array.push(tkill_a);
        tm_array.push(tkill_b);
        tm_array.push(tkill_c);

        let elapsed = now.elapsed();
        let sec = (elapsed.as_secs() as f64) + (elapsed.subsec_nanos() as f64 / 1000_000_000.0);
        println!("Seconds: {}", sec);
        println!("");
    }
}

fn main() {
    let mut rng = rand::thread_rng();

    multi_threads();

    //digit_tests(&mut rng);

    println!("Hello, world!");

    //Ok(())
}
