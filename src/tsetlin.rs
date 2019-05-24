
#![allow(unused_parens)]
#![allow(non_snake_case)]

use std::process;
use rand::prelude::*;
use std::option::Option;

pub const INT_SIZE: u64 = 64;

#[derive(Clone)]
pub struct TsetlinSub {
    pub ta_state: Vec<u64>
    ,pub clause_output: Vec<u64>
    ,pub feedback_to_la: Vec<u64>
    ,pub feedback_to_clauses: Vec<u64>
    ,pub prob_svalues: Vec<f32>
    ,pub threshold: u64
    ,pub features: u64
    ,pub clauses: usize
    ,pub state_size: usize
    ,pub clause_chunks: usize
    ,pub classes: usize
    ,pub filter: u64
    ,pub state_bits: usize
    ,pub la_chunks: usize
}

#[derive( Clone)]
pub struct TsetlinMachine {
    pub tclasses: Vec<TsetlinSub>
    ,pub classes: usize
    ,pub features: u64
    ,pub la_chunks: usize
    ,pub score: Vec<f64>
    ,pub id_num: u64
    ,pub life: u64
    ,pub svalue: f32
}


pub fn copy_tm(t: &mut TsetlinMachine) -> TsetlinMachine{
    let mut my_vec = Vec::with_capacity(t.tclasses.len());

    for i in 0..t.tclasses.len() {
        my_vec.push(TsetlinSub {
            ta_state: t.tclasses[i].ta_state.to_vec()
            ,
            clause_output: t.tclasses[i].clause_output.to_vec()
            ,
            feedback_to_la: t.tclasses[i].feedback_to_la.to_vec()
            ,
            feedback_to_clauses: t.tclasses[i].feedback_to_clauses.to_vec()
            ,
            prob_svalues: t.tclasses[i].prob_svalues.to_vec()
            ,
            threshold: t.tclasses[i].threshold
            ,
            features: t.tclasses[i].features
            ,
            clauses: t.tclasses[i].clauses
            ,
            state_size: t.tclasses[i].state_size
            ,
            clause_chunks: t.tclasses[i].clause_chunks
            ,
            classes: t.tclasses[i].classes
            ,
            filter: t.tclasses[i].filter
            ,
            state_bits: t.tclasses[i].state_bits
            ,
            la_chunks: t.tclasses[i].la_chunks
        });
    }

    let mut tnew: TsetlinMachine = TsetlinMachine{
        tclasses: my_vec
        ,classes: t.classes
        ,features: t.features
        ,la_chunks: t.la_chunks
        ,score: vec![0.0; 3]
        ,id_num: t.id_num
        ,life: 0
        ,svalue: t.svalue
    };

    tnew
}


#[inline]
fn d3(t: &mut TsetlinSub, j: usize, k: usize, b: usize) ->usize {
    (j * t.la_chunks * t.state_bits) + (k * t.state_bits) + b
}

//Should find binomial rust code but this will only run once on setup so it should be a one time hit.
fn calc_prob_array(svalue: f32, prob_svalues: &mut Vec<f32>){
    let mut rng = rand::thread_rng();
    let mut count_array: Vec<u64> = vec![0; 32];
    let number_tests: u64 = 1_222_000;

    for i in 0..number_tests{
        let mut count: usize = 0;
        for j in 0..64{
            let rfl: f32 = rng.gen();
            //println!(" rfl: {}", rfl);
            if(rfl < 1.0/svalue){
                if(count >= 32){
                    break;
                }
                count_array[count] += 1;
                count += 1;
            }
        }
    }

    for i in 0..prob_svalues.len(){
        prob_svalues[i] = (count_array[i] as f32) / (number_tests as f32);
        //println!(" probability of i:{}  prob:{:.6}  count_array:{}  svalue:{}",i,prob_svalues[i],count_array[i], svalue);
        if(prob_svalues[i] < 0.000001){
            break;
        }
        if(i == prob_svalues.len() - 1){
            prob_svalues[i] = -1.0;
        }
    }

}

pub fn test_d3(tm_sub: &mut TsetlinSub){
    let mut ta: Vec<u64> =vec![0; tm_sub.state_size];
    for j in 0..tm_sub.clauses{
        for k in 0..tm_sub.la_chunks{
            for b in 0..tm_sub.state_bits{
                let idx = d3(tm_sub, j,k,b);
                if(ta[idx] != 0){
                    println!("ERROR: d3 found value already written");
                    process::exit(0);
                }
                ta[idx] = 1234;
            }
        }
    }

    for idx in 0..tm_sub.state_size{
        if(ta[idx] != 1234){
            println!("ERROR: d3 not all values in three dimensional array written");
            process::exit(0);
        }
    }
    println!("Passed: d3 test");
}



pub fn init_tm(
    threshold: u64
    , features: u64
    , clauses: usize
    , classes: usize
    , state_bits: usize
    , id_num: u64
    , svalue: f32
) -> TsetlinMachine {
    let la_chunks: usize = (((2 * features) - 1)/INT_SIZE + 1) as usize;

    let mut prob_svalues: Vec<f32> = vec![0.0; 32];

    calc_prob_array(svalue,&mut prob_svalues);

    //println!(" ----->  out initialized prob_svalues:{}", prob_svalues[0]);

    let mut tm = TsetlinMachine{
        tclasses: Vec::with_capacity(classes)
        //,prob_svalues: prob_svalues.to_vec()
        ,classes: classes
        ,features: features
        ,la_chunks: la_chunks
        ,score: vec![0.0; 3]
        ,id_num: id_num
        ,life: 0
        ,svalue: svalue
    };

    for _ in 0..classes{
        tm.tclasses.push(init_tm_sub(threshold, features, clauses, classes, state_bits, la_chunks, prob_svalues.to_vec()) );
    }

    tm
}

pub fn breed(t1:& TsetlinMachine, t2: & TsetlinMachine, kill: &mut TsetlinMachine, rng: &mut ThreadRng){
    kill.life = 0;
    kill.score = vec![0.0; 3];

    let mut r: u64 = rng.gen();

    for s in 0..kill.tclasses.len() {
        for j in 0..kill.tclasses[s].clauses {
            r = rng.gen();
            for k in 0..kill.tclasses[s].la_chunks{
                if (r % 2 == 0) {
                    //kill.tclasses[s].  la_chunks[k] = t1.tclasses[s].la_chunks[k];
                    kill.tclasses[s].feedback_to_la[k] = t1.tclasses[s].feedback_to_la[k];
                } else {
                    //kill.tclasses[s].la_chunks[k] = t2.tclasses[s].la_chunks[k];
                    kill.tclasses[s].feedback_to_la[k] = t2.tclasses[s].feedback_to_la[k];
                }
                for b in 0..kill.tclasses[s].state_bits{
                    let idx = d3( &mut kill.tclasses[s], j, k, b);
                    if(r % 2 == 0) {
                        kill.tclasses[s].ta_state[idx] = t1.tclasses[s].ta_state[idx];
                    } else {
                        kill.tclasses[s].ta_state[idx] = t2.tclasses[s].ta_state[idx];
                    }
                }
            }
        }
    }
}

pub fn breed_old(t1:& TsetlinMachine, t2: & TsetlinMachine, kill: &mut TsetlinMachine, rng: &mut ThreadRng){
    kill.life = 0;
    kill.score = vec![0.0; 3];

    let mut r: u64 = rng.gen();

    for t_idx in 0..kill.tclasses.len() {
        for i in 0..kill.tclasses[0].ta_state.len() {
            if(i % (kill.tclasses[0].feedback_to_la.len() * 64) == 0){
                r = rng.gen();
            }

            if (i < kill.tclasses[t_idx].ta_state.len()) {
                if (r % 2 == 0) {
                    kill.tclasses[t_idx].ta_state[i] = t1.tclasses[t_idx].ta_state[i];
                } else {
                    kill.tclasses[t_idx].ta_state[i] = t2.tclasses[t_idx].ta_state[i];
                }
            }

            if (i < kill.tclasses[t_idx].feedback_to_la.len()) {
                if (r % 2 == 0) {
                    kill.tclasses[t_idx].feedback_to_la[i] = t1.tclasses[t_idx].feedback_to_la[i];
                } else {
                    kill.tclasses[t_idx].feedback_to_la[i] = t2.tclasses[t_idx].feedback_to_la[i];
                }
            }

            if (i < t1.tclasses[t_idx].feedback_to_clauses.len()){
                if (r % 2 == 0) {
                    kill.tclasses[t_idx].feedback_to_clauses[i] = t1.tclasses[t_idx].feedback_to_clauses[i];
                } else {
                    kill.tclasses[t_idx].feedback_to_clauses[i] = t2.tclasses[t_idx].feedback_to_clauses[i];
                }
            }
        }
    }
}

fn init_tm_sub(
    threshold: u64
    , features: u64
    , clauses: usize
    , classes: usize
    , state_bits: usize
    , la_chunks: usize
    , prob_svalues: Vec<f32>
) -> TsetlinSub {
    let clause_chunks: usize = (((clauses-1) as u64)/INT_SIZE + 1) as usize;
    let filter = (!(0xffffffffffffffff << ((features*2) % INT_SIZE)));
    let state_size: usize = clauses * la_chunks * state_bits;

    //println!(" out initialized prob_svalues:{}", prob_svalues[0]);

    let mut t = TsetlinSub
        {
            ta_state: vec![0; state_size]
            , clause_output: vec![0; clause_chunks]
            , feedback_to_la: vec![0; la_chunks]
            , feedback_to_clauses: vec![0; clause_chunks]
            , prob_svalues: prob_svalues.to_vec()  //Copy of probability array
            , threshold: threshold
            , features: features
            , clause_chunks: clause_chunks
            , state_size: state_size
            , clauses: clauses
            , classes: classes
            , filter: filter
            , state_bits: state_bits
            , la_chunks: la_chunks
        };

    for j in 0..clauses{
        for k in 0..la_chunks{
            for b in 0..state_bits - 1{
                let idx = d3(&mut t,j,k,b);
                t.ta_state[idx] = !(0 as u64);
                //println!(" hi  {:x}",t.ta_state[d3(j,k,b)])
            }
            let idx = d3(&mut t,j,k,state_bits - 1);
            t.ta_state[idx] = 0;
        }
    }

    t
}

fn tm_calculate_clause_output(tm_sub: & mut TsetlinSub, Xi: & Vec<u64>, predict: bool){
    //tm_calculate_clause_output
    for j in tm_sub.clause_output.iter_mut() {
        *j = 0;
    }

    //let mut count_yes : i32 = 0;
    //let mut count_no: i32 = 0;

    for j in 0..tm_sub.clauses{
        let mut output:bool = true;
        let mut all_exclude:bool = true;
        for k in 0..tm_sub.la_chunks-1{
            //println!(" k:{}  x:{}",k,Xi[k]);
            let idx = d3(tm_sub, j,k, tm_sub.state_bits - 1);
            let val = tm_sub.ta_state[idx];
            //println!(" k:{}   x:{}   val:{:x}",k,Xi[k],val);
            //println!(" val:{}",val);
            output = output && (val & Xi[k]) == val;

            if !output {
                break;
            }
            all_exclude = all_exclude && (val == 0);
        }

        let idx = d3(tm_sub,j,tm_sub.la_chunks-1,tm_sub.state_bits-1);
        let val = tm_sub.ta_state[idx];

        output = output &&
            val & Xi[tm_sub.la_chunks-1] & tm_sub.filter == val & tm_sub.filter ;

        all_exclude = all_exclude && (val == 0);

        output = output && !(predict && all_exclude);

        if(output){
            //count_yes += 1;
            let clause_chunk = j / (INT_SIZE as usize);
            let clause_chunk_pos = j % (INT_SIZE as usize);
            tm_sub.clause_output[clause_chunk] |= (1 << clause_chunk_pos);
        } else {
            //count_no += 1;
        }
    }
    //println!(" y{}  n{}",count_yes,count_no);
}


fn tm_update(tm_sub: & mut TsetlinSub, Xi: & Vec<u64>, target: u64, rng: &mut  ThreadRng){

    tm_calculate_clause_output(tm_sub, Xi, false);

    let class_sum = sum_up_class_votes(tm_sub);

    //println!(" class_sum:{}",class_sum);

    let threshold: f64 = (1.0/(tm_sub.threshold as f64 *2.0)) * (tm_sub.threshold as f64 + (1.0 - 2.0*(target as f64)) * (class_sum as f64));

    if(threshold > 0.49999 && threshold < 0.50001){
        for j in tm_sub.feedback_to_clauses.iter_mut() {
            *j = rng.gen();
            //println!("  tm_sub.feedback_to_clauses[j]:{}",tm_sub.feedback_to_clauses[j]);
        }
    } else {
        for j in tm_sub.feedback_to_clauses.iter_mut(){
            *j = 0;
        }

        for j in 0..tm_sub.clauses{
            let clause_chunk = j/(INT_SIZE as usize);
            let clause_chunk_pos = j % (INT_SIZE as usize);
            let fl: f64 = rng.gen();
            //println!(" fl:{}",fl);
            if(fl <= threshold) {
                tm_sub.feedback_to_clauses[clause_chunk] |= (1 << clause_chunk_pos);
            }
        }
    }

    for j in 0..tm_sub.clauses{
        let j_i64: i64 = j as i64;
        let clause_chunk = j/(INT_SIZE as usize);
        let clause_chunk_pos = j % (INT_SIZE as usize);

        let bit_set = (tm_sub.feedback_to_clauses[clause_chunk]) & (1 << clause_chunk_pos) > 0;

        if(! bit_set){
            continue;
        }

        if (2 * (target as i64) - 1) * (1 - 2 * (j_i64 & 1)) == -1 {
            if (tm_sub.clause_output[clause_chunk] & (1 << clause_chunk_pos)) > 0 {
                for k in 0..tm_sub.la_chunks {
                    //println!("got here");
                    let idx = d3(tm_sub, j,k, tm_sub.state_bits - 1);
                    tm_inc(tm_sub,j,k, (!Xi[k]) & (!(tm_sub.ta_state[idx])));
                }
            }
        } else if ((2 * (target as i64) - 1) * (1 - 2 * (j_i64 & 1)) == 1){
            //println!("hummmm");
            tm_initialize_random_streams(tm_sub, rng);

            if (tm_sub.clause_output[clause_chunk] & (1 << clause_chunk_pos)) > 0 {
                for k in 0..tm_sub.la_chunks {
                    tm_inc(tm_sub,j,k, Xi[k]);

                    tm_dec(tm_sub, j,k, (!Xi[k]) & tm_sub.feedback_to_la[k]);
                }
            } else {
                for k in 0..tm_sub.la_chunks {
                    tm_dec(tm_sub, j, k, tm_sub.feedback_to_la[k]);
                }
            }
        }

    }
}


pub fn tm_fit(tm: & mut TsetlinMachine, x_vec: & Vec<Vec<u64>>, y_vec: & Vec<u64>, number_of_examples: usize, rng: &mut ThreadRng){
    //let mut rng = rand::thread_rng();

    //println!("starting examples processing");
    for i in 0..number_of_examples {
        //let mut x = x_vec[i];
        let target_class = y_vec[i];
        let mut negative_target_class: u64;

        //println!("zero element:{}",(x_vec[i])[0]);
        tm_update(&mut tm.tclasses[target_class as usize], & x_vec[i], 1, rng);

        if(tm.classes == 2){
            if(target_class == 1) {
                negative_target_class = 0;
            } else {
                negative_target_class = 1;
            }
        } else {
            negative_target_class = rng.gen();
            negative_target_class = negative_target_class % (tm.classes as u64);
            while negative_target_class == target_class {
                negative_target_class = rng.gen();
                negative_target_class = negative_target_class % (tm.classes as u64);
            }
        }
        //println!(" target_class:{}   negative_target_class:{}",target_class,negative_target_class);

        tm_update(&mut tm.tclasses[negative_target_class as usize], & x_vec[i], 0,  rng);
    }
}

fn sum_up_class_votes(tm_sub: &mut TsetlinSub) -> i64{
    let mut class_sum: i64 = 0;

    for j in tm_sub.clause_output.iter_mut(){
        let output: u64 = *j;
        //println!(" output:{}",output);

        let plus: u64 = output & 0x5555555555555555;
        let minus: u64 = output & 0xaaaaaaaaaaaaaaaa;

        let p1:u32 = plus as u32;
        let p2:u32 = (plus >> 32) as u32;

        let m1: u32 = minus as u32;
        let m2: u32 = (minus >> 32) as u32;

        class_sum += (p1.count_ones() + p2.count_ones()) as i64;
        class_sum -= (m1.count_ones() + m2.count_ones()) as i64;
    }

    class_sum = if(class_sum > tm_sub.threshold as i64) { tm_sub.threshold as i64} else { class_sum };
    class_sum = if(class_sum < -(tm_sub.threshold as i64)) { -(tm_sub.threshold as i64)} else { class_sum };

    //println!(" class_sum:{}",class_sum);

    class_sum
}

fn tm_inc(tm_sub: &mut TsetlinSub, clause: usize, chunk: usize, active: u64){
    let mut carry = active;

    for b in 0..tm_sub.state_bits{
        if(carry == 0){
            break;
        }

        let idx = d3(tm_sub,clause,chunk,b);
        let carry_next = tm_sub.ta_state[idx] & carry;
        tm_sub.ta_state[idx] = tm_sub.ta_state[idx] ^ carry;
        carry = carry_next;
    }

    if(carry > 0){
        for b in 0..tm_sub.state_bits{
            let idx = d3(tm_sub,clause,chunk,b);
            tm_sub.ta_state[idx] |= carry;
        }
    }
}

fn tm_dec(tm_sub: &mut TsetlinSub, clause: usize, chunk: usize, active: u64){
    let mut carry = active;

    for b in 0..tm_sub.state_bits{
        if(carry == 0){
            break;
        }

        let idx = d3(tm_sub, clause,chunk,b);
        let carry_next = (!tm_sub.ta_state[idx]) & carry;
        tm_sub.ta_state[idx] = tm_sub.ta_state[idx] ^ carry;
        carry = carry_next;
    }

    if(carry > 0){
        for b in 0..tm_sub.state_bits{
            let idx = d3(tm_sub,clause,chunk,b);
            tm_sub.ta_state[idx] &= (!carry);
        }
    }
}

fn tm_initialize_random_streams(tm_sub: &mut TsetlinSub, rng: &mut ThreadRng){

    for k in 0..tm_sub.la_chunks{
        tm_sub.feedback_to_la[k] = 0;
        let mut rfl:f32 = rng.gen();
        let mut rnext: usize = 0;
        /********************************************
          The vector prob_svalues has a crudely calculated bionomial probability that a bit is set in our 64bit value.
           So for example if svalue = 10 then:
            prob_svalues[0] =  .9988
            prob_svalues[1] =  .9904
            prob_svalues[2] =  .9610
            prob_svalues[3] =  .8937
            prob_svalues[4] =  .7795
            prob_svalues[5] =  .6272
            ...
        ********************************************/
        while (rfl < tm_sub.prob_svalues[rnext]) {

            //println!(" rfl:{}  rnext:{}  prob_svalues:{}",rfl,rnext,tm_sub.prob_svalues[rnext]);
            let mut r: u32 = rng.gen();
            r = r & 0x00000003f;  // equal to r % 64
            //We could end up setting the same bit over but I don't want to take the speed hit in order to check.
            tm_sub.feedback_to_la[k] |= (1 << r);

            rfl = rng.gen();
            rnext += 1;
        }
    }
}

pub fn tm_score(tm_sub: &mut TsetlinSub, x: & Vec<u64>) ->i64 {

    tm_calculate_clause_output(tm_sub, x, true);

    sum_up_class_votes(tm_sub)
}

pub fn binary_score(tm: &mut TsetlinMachine, x: &mut Vec<u64>) -> u64 {
    let max_class_sum = tm_score(&mut tm.tclasses[0], x);

    let class_sum = tm_score(&mut tm.tclasses[1], x);
    if max_class_sum < class_sum {
        1
    } else {
        0
    }
}

pub fn tm_eval(tm: &mut TsetlinMachine, x_vec: & Vec<Vec<u64>>, y_vec: & Vec<u64>, number_of_examples: usize, score_idx: usize) -> f64{
    let mut errors: i64;
    let mut max_class: u64;
    let mut max_class_sum: i64;
    let mut class_sum: i64 = 0;

    errors = 0;
    for l in 0..number_of_examples{
        max_class_sum = tm_score(&mut tm.tclasses[0], & x_vec[l]);
        //print!(" {}",max_class_sum);
        max_class = 0;
        for i in 1..tm.classes{
            class_sum = tm_score(&mut tm.tclasses[i], & x_vec[l]);
            //print!(" {}",class_sum);
            if max_class_sum < class_sum {
                max_class_sum = class_sum;
                max_class = i as u64;
            }
        }

        //println!(" max_class:{}     max_class_sum:{} ",max_class,max_class_sum);

        if max_class != y_vec[l]{
            errors += 1;
        }
    }

    tm.score[score_idx] = 1.0 - ((errors as f64)/(number_of_examples as f64));

    tm.score[score_idx]
}