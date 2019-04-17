from distutils.core import setup

# from setuptools import setup

setup(name='text_mining',
      version='1.4',
      script_name='Text_Mining_Pipeline_Starter.py',
      data_files=[('config', ['config/config.properties']),
                  ('data', ['model_vectorizer_pickles/tfidf_vect.pkl', 'model_vectorizer_pickles/trained_model.pkl'])],
      packages=['text_mining', 'text_mining.src', 'text_mining.src.utility'],
      author='Manu_Chauhan',
      install_requires=['scikit-learn', 'pattern3', 'psycopg2', 'nltk', 'numpy', 'pandas', 'stop_words',
                        'guess_language-spirit'],
      include_package_data=True
      )
