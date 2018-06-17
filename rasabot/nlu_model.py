from rasa_nlu import config
from rasa_nlu.training_data import load_data
from rasa_nlu.model import Trainer, Interpreter


def train_nlu(data, configs, model_dir):
    """
        Train a NLU model

        :param data:
        :param configs:
        :param model_dir:
        :return: None
    """
    training_data = load_data(data)
    trainer = Trainer(config.load(configs))
    trainer.train(training_data)
    model_directory = trainer.persist(model_dir, fixed_model_name='weathernlu')


def run_nlu():
    """
        Test the trained NLU model
        :return: None
    """
    interpreter = Interpreter.load('./model/nlu/default/weathernlu')
    print(interpreter.parse('Going to London next week. What is the weather there?'))


if __name__ == '__main__':
    train_nlu('./data/data.json', './config_spacy.json', './model/nlu')
    run_nlu()

