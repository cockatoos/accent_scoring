# 04-run-pytorch.py
from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core import Environment
from azureml.core import ScriptRunConfig
from azureml.core import Dataset
from azureml.core.model import Model
if __name__ == "__main__":
    ws = Workspace.from_config()
    experiment = Experiment(workspace=ws, name='day1-experiment-train')
    orig_dataset = Dataset.get_by_name(workspace=ws, name='recordings')
    train_dataset = Dataset.get_by_name(workspace=ws, name='train')
    test_dataset = Dataset.get_by_name(workspace=ws, name='test')
    val_dataset = Dataset.get_by_name(workspace=ws, name='val')


    # Tip: When model_path is set to a directory, you can use the child_paths parameter to include
    #      only some of the files from the directory
    model = Model.register(model_path = "./models",
                           model_name = "accent_detection",
                           description = "distinguish native English accent from foreign accents",
                           workspace = ws)

    config = ScriptRunConfig(source_directory='./src',
                             script='AccentClassification.py',
                             compute_target='model-training-machine',
                             arguments=[
                                '--orig_data_path', orig_dataset.as_mount(),
                                '--train_data_path', train_dataset.as_mount(),
                                '--test_data_path', test_dataset.as_mount(),
                                '--val_data_path', val_dataset.as_mount(),
                                '--model_path', './models'
                             ])
    # set up pytorch environment
    env = Environment.from_pip_requirements(
        name='accent_scoring_env',
        file_path='/Users/yejinseo/Desktop/azure_ai_hack/accent_scoring/requirements.txt'
    )
    # Creates the environment inside a Docker container.
    env.docker.enabled = True
    # Specify docker steps as a string.
    dockerfile = r'''
    FROM mcr.microsoft.com/azureml/intelmpi2018.3-ubuntu16.04
    RUN echo "Hello from custom container!"
    RUN apt-get install -y libsndfile1
    '''

    env.docker.base_dockerfile = dockerfile

    config.run_config.environment = env

    run = experiment.submit(config)
    run.wait_for_completion(show_output=True)
    aml_url = run.get_portal_url()
    print(aml_url)
