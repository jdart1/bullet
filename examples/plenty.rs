use bullet_lib::{
    inputs, loader, lr, optimiser, outputs,
    testing::{Engine, EngineType, OpeningBook, TestSettings, TimeControl, UciOption},
    wdl, Activation, LocalSettings, Loss, TrainerBuilder, TrainingSchedule, TrainingSteps,
};

use std::{
    path::PathBuf,
    process::Command,
    io,
};

use errno::errno;

macro_rules! net_id {
    () => {
        "plenty"
    };
}

const NET_ID: &str = net_id!();

fn main() {
    #[rustfmt::skip]
    let mut trainer = TrainerBuilder::default()
        .quantisations(&[510, 64])
        .optimiser(optimiser::AdamW)
        .loss_fn(Loss::SigmoidMSE)
        .input(inputs::ChessBucketsMirrored::new([
        0, 0, 1, 1,
        2, 2, 2, 2,
        3, 3, 3, 3,
        4, 4, 4, 4,
        5, 5, 5, 5,
        5, 5, 5, 5,
        6, 6, 6, 6,
        6, 6, 6, 6
        ]))
        .output_buckets(outputs::MaterialCount::<8>)
        .feature_transformer(2048)
        .activate(Activation::SCReLU)
        .add_layer(1)
        .build();

    let schedule = TrainingSchedule {
        net_id: NET_ID.to_string(),
        eval_scale: 400.0,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: 200,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 0.0 },
        lr_scheduler: lr::StepLR { start: 0.001, gamma: 0.3, step: 60 },
        save_rate: 150,
    };

    let optimiser_params =
        optimiser::AdamWParams { decay: 0.01, beta1: 0.9, beta2: 0.999, min_weight: -1.98, max_weight: 1.98 };

    trainer.set_optimiser_params(optimiser_params);

    let settings = LocalSettings { threads: 8,
       //test_set: Option::Some(TestDataset.new("/data2/bullet/sep2024/validationdata/val1.bullet",20)),
       test_set: None,
       output_directory: "checkpoints", batch_queue_size: 512 };

    let data_loader = loader::DirectSequentialDataLoader::new(&["/data2/bullet/oct2024/new/trainingdata/pos1.bullet"]);

    pub struct ArasanEngine;

    impl EngineType for ArasanEngine {
      fn build(&self, inp_path: &str, out_path: &str, net: Option<&str>) -> Result<(), String> {
          let mut submodule = Command::new("git");

          submodule.current_dir(inp_path).
             args(["submodule","update","--init","--recursive"]).
             output().
             expect("failed to execute git submodule");

          let mut build_base = Command::new("make");

          // Arasan makefile is in src subdir
          let path : PathBuf = [inp_path, "src"].iter().collect();

          // out path is relative to repo dir, but we will cd one level lower, so correct here
          let out_path2 : PathBuf = ["..", out_path].iter().collect();
          let out_path_str = out_path2.to_str().unwrap();

          build_base.current_dir(path).arg(format!("EXE={out_path_str}")).arg("CC=clang").arg("BUILD_TYPE=avx2").arg("-j");

          if let Some(net_path) = net {
              build_base.arg(format!("NETWORK={}", net_path));
          }

          match build_base.output() {
              io::Result::Err(err) => Err(format!("Failed to build engine: {err}!")),
              io::Result::Ok(out) => {
                  if out.status.success() {
                      Ok(())
                  } else {
                      println!("{}", String::from_utf8(out.stdout).unwrap());
                      Err(String::from("Failed to build engine!"))
                  }
              }
          }
       }

       fn bench(&self, path: &str) -> Result<usize, String> {
          println!("running bench on exe path {path}");
          let mut bench_cmd = Command::new(path);

          let output = bench_cmd.arg("bench").output();

          let mut bench_string = String::from("");

          match output {
              io::Result::Err(err) => {
                  let e = errno();
                  println!("Error {}: {}", e, err);
              },
              io::Result::Ok(out) => {
                  assert!(out.status.success(), "Bad status from bench run!");
                  bench_string = String::from_utf8(out.stdout).expect("Could not parse bench output!");
              }
          };

          let split = bench_string.split_whitespace();

          let mut bench = None;

          let mut idx : u32 = 0;
          let mut target : u32 = 10000000;
          for word in split {
               if word == "Nodes" {
                  target = idx + 2;
              }
              if idx == target {
                 bench = word.parse().ok();
                 break;
              }
              idx = idx + 1;
          }

          if let Some(bench) = bench {
              Ok(bench)
          } else {
              Err(String::from("Failed to parse bench output!"))
          }
       }
    }

    let base_engine = Engine {
        repo: "https://github.com/jdart1/arasan-chess",
        branch: "test",
        bench: None,
        net_path: None,
        uci_options: vec![UciOption("Hash", "16")],
        engine_type: ArasanEngine,
    };

    let dev_engine = Engine {
        repo: "https://github.com/jdart1/arasan-chess",
        branch: "test",
        bench: None,
        net_path: None,
        uci_options: vec![UciOption("Hash", "16")],
        engine_type: ArasanEngine,
    };

    let testing = TestSettings {
        test_rate: 20,
        out_dir: concat!("../../nets/", net_id!()),
        cutechess_path: "/home/jdart/chess/cutechess-cli/cutechess-cli",
        book_path: OpeningBook::Pgn("/home/jdart/chess/books/8moves_v3.pgn"),
        num_game_pairs: 2000,
        concurrency: 6,
        time_control: TimeControl::FixedNodes(25_000),
        base_engine,
        dev_engine,
    };

    trainer.run_and_test(&schedule, &settings, &data_loader, &testing);

    for fen in [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    ] {
        let eval = trainer.eval(fen);
        println!("FEN: {fen}");
        println!("EVAL: {}", 400.0 * eval);
    }
}
