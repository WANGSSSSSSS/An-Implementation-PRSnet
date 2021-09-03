import argparse
from core.context import Context

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--train_data", default="")
    args.add_argument("--test_data",  default="")
    args.add_argument("--lr", default=0.001)
    args.add_argument("--batch_size", default=2)
    args.add_argument("--epochs", default=100)
    args.add_argument("--sample_num", default=10)
    args.add_argument("--predict_num", default=3)
    args.add_argument("--save_path", default="")
    args.add_argument("--log_path", default="")
    args.add_argument("--alpha", default=32)
    args.add_argument("--load_path", default=None, type=str)
    args.add_argument("--train", default=True, type=bool)
    args = args.parse_args()

    controller = Context(
        SN=args.sample_num,
        PN=args.predict_num,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_dir=args.save_path,
        log_dir=args.log_path,
        lr=args.lr,
        alpha=args.alpha,
        train=args.train
    )
    if args.load_path is not None:
        controller.load_context(args.load_path)


    controller()

