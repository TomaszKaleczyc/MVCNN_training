import argparse

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer

from dataset.mvcnn_data_module import MVCNNDataModule
from model.mvcnn import MVCNNClassifier
from model.callbacks import UnfreezePretrainedWeights


def mvcnn_argparser(epilog=None):
    """
    CLI arguments used for model training
    """
    parser = argparse.ArgumentParser(
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--NUM_CLASSES', type=int, default=None, help='Number of classes used in training')
    parser.add_argument('--LEARNING_RATE', type=float, default=1e-3, help='Model initial learning rate')
    parser.add_argument('--LEARNING_RATE_REDUCTION_FACTOR', type=float, default=1e3, 
                        help='Factor by which the learning rate is divided after the feature extractor weights are unfrozen')
    parser.add_argument('--NUM_EPOCHS', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--NUM_EPOCHS_FREEZE_PRETRAINED', type=int, default=20, 
                        help='Number of training epochs with frozen feature extractor weights')
    parser.add_argument('--BATCH_SIZE', type=int, default=1, help='Number of instances used in a training step')
    parser.add_argument('--DROPOUT_RATE', type=float, default=0.3, help='Probability of using dropout in the model layers')
    parser.add_argument('--SAVE_PATH', type=str, default='./output', help='Directory where training output is saved')
    return parser


if __name__ == '__main__':
    args = mvcnn_argparser().parse_args()
    print('Parsed arguments:', args)

    data_module = MVCNNDataModule(
        args.NUM_CLASSES, 
        args.BATCH_SIZE
        )

    model = MVCNNClassifier(
        num_classes=args.NUM_CLASSES,
        learning_rate=args.LEARNING_RATE,
        num_epochs_freeze_pretrained=args.NUM_EPOCHS_FREEZE_PRETRAINED,
        dropout_rate=args.DROPOUT_RATE,
        )

    callbacks = [
        ModelCheckpoint(
            filename='mvcnn-e{epoch:.03d}-val_f1{validation/f1:.3f}',
            monitor='validation/f1', 
            save_top_k=3,
            verbose=True, 
            mode='max'
            ),
        UnfreezePretrainedWeights(args.LEARNING_RATE_REDUCTION_FACTOR),
    ]

    trainer = Trainer(
        max_epochs=args.NUM_EPOCHS,
        fast_dev_run=False,
        default_root_dir=args.SAVE_PATH,
        callbacks=callbacks
    )

    trainer.fit(
        model,
        train_dataloader=data_module.train_dataloader(),
        val_dataloaders=data_module.val_dataloader()
    )