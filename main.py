from face import Capture, Recognition, Inference, Tracking

if __name__ == '__main__':
    folder_name = 'training_data'
    num_people = int(input("Number of people: "))
    num_images = int(input("Images per person: "))
    
    image_capture = Capture(folder_name)
    
    for i in range(num_people):
        target_name = input(f"Enter a label for person {i + 1}: ")
        image_capture.capture_images(target_name, num_images)
    
    image_capture.write_to_csv()

    trainer = Recognition(data_path='training_data', weights_path='code.pt')
    trainer.load_model()
    trainer.print_model()
    trainer.prepare_data_loaders()
    trainer.train_model()
    trainer.save_model()

    inference = Inference(data_path='training_data', model_path='code.pt')
    inference.infer_faces()

    tracking = Tracking(data_path='training_data', model_path='code.pt')
    tracking.load_model()
    tracking.track_and_recognize_faces()
