import mlflow

try:
    # Creating an experiment 
    mlflow.create_experiment('demo_data_process_flow')
except:
    pass
# Setting the environment with the created experiment
mlflow.set_experiment('demo_data_process_flow')


def run_model(**kwargs):
    '''
    Run your modelling tasks here
    '''
    pass
