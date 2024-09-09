if __name__ == '__main__':

    import torch
    import os
    import data_setup, engine, model_builder, utils
    import argparse

    from torchvision import transforms

    parser = argparse.ArgumentParser(description="Get some hyperparameters")

    parser.add_argument("--num_epochs",
                        default=10,
                        type=int,
                        help="the number of epochs to train for")
    parser.add_argument("--batch_size",
                        default=32,
                        type=int,
                        help="number of sample per batch")
    parser.add_argument("--hidden_units",
                        default=10,
                        type=int,
                        help="number of hidden units in hidden layer")
    parser.add_argument("--lr",
                        default=0.001,
                        type=float,
                        help="learning reate to use for model")

    args = parser.parse_args()

    NUM_EPOCHS = args.num_epochs
    BATCH_SIZE = args.batch_size
    HIDDEN_UNITS = args.hidden_units
    LEARNING_RATE = args.lr

    train_dir = "./data/pizza_steak_sushi/train"
    test_dir = "./data/pizza_steak_sushi/test"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor()
    ])

    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                                    test_dir=test_dir,
                                                                                    transform=data_transform,
                                                                                    batch_size=BATCH_SIZE)

    model = model_builder.TinyVGG(input_shape=3,
                                    hidden_units=HIDDEN_UNITS,
                                    output_shape=len(class_names)).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    engine.train(model=model,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                epochs=NUM_EPOCHS,
                device=device)
        
    utils.save_model(model=model,
                    target_dir="models",
                    model_name="first_going_modular_model.pth")


