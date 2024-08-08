## SALT 🧂 Simple Active-Learning Tool

SALT is a Streamlit app for textual data analysis 📚🔬

Just give it a dataset of texts, and it will allow you to:

* Perform semantic-search over the dataset 🔎
* Label examples efficiently through active-learning 🔄
* Create a simple & fast (yet surprisingly good) classifier 🤖
* Find clusters of similar examples (lexically and/or semantically) 🗄️


## How to run?
1. Clone the repository
```
git clone https://github.com/AI21Labs/salt.git
cd salt
```

2. (Optional) Set up virtual environment
```
pyenv virtualenv 3.9.0 salt
pyenv activate salt
pip install --upgrade pip
```

3. Install python dependencies
```
pip install -r requirements.txt
```

4. Run the app
```
python -m streamlit run salt/view/main.py
```

## How to use?
Here is the basic flow (for more advanced options see the FAQ section below)

#### Create a project ⚙️️
1. Load a CSV or JSON-lines file
2. Select the relevant column and choose a name for the project
3. Click on "Create project" and wait until the creation process is completed

#### (Optional) Find clusters 🗄️
1. Go to the "Clusters" step
2. Choose similarity type and number of clusters
3. Click on "Run clustering" and wait for the clustering process to finish
4. Review the results (clustering overview / by cluster)

#### Active-Learning 🔄
1. Go to the "Review 📖" step
2. Define the classes: provide at least one "seed example" for each class
    * You can use "Search 🔎" to easily find relevant examples
    * Assign each example with its label by editing the "label" column
3. Go to the "Labeling 🖊️️" step
4. Label examples one by one (chosen by the active-learning classifier)
5. After labeling 10 examples, you'll start to see an updating graph of predictions change-rate
    * It may help you know when to stop (once each class has converged to some stable mode)
6. At any point, you can go back to "Review 📖" to:
    * Download all labels and predictions
    * View labels/predictions for specific examples
    * Add a new class (by providing its "seed example")


## FAQ
#### 1. Can I run the classifier on new examples?
Yes! Go to the "Inference 🔦" step, which provides several options for running the classifier:

* Insert any text 🔤
* Upload a file of texts 📃
* From code 💻 (export the model and use the code sample to run it)

#### 2. Can I add new examples to the project?
Yes! This can be done by creating a new project that extends the current one:

1. Go to the "Setup ⚙️️" step, upload the new examples file, select the relevant column and choose the project name
2. Click on "Optional settings", select the base project and then click on "Create project"

#### 3. Can I start from a dataset with some already-labeled examples?
Yes! When you create the project, click on "Optional settings" and select the label column

#### 4. Can I do multi-label classification?
Yes! If you insert a seed example with multiple labels (separated by a comma, e.g. "pos,neg"),
your classifier and the labeling interface mode will turn from "Single-label 📎️" into "Multi-label 🖇️️"

#### 5. My classifier is not good enough. What can I do?
So you've labeled some examples, and now when you look at the predictions you see that a lot of them are wrong.
To improve the classifier, you can try one of the following:

* Go to the "Labeling 🖊️️" step and keep labeling. More data is always better
* Go to the "Review 📖️️" step, find some wrong predictions and insert their correct labels. Then make a few more "Labeling 🖊️️" iterations to stabilize the classifier
* Create a new project with a simpler version of the texts, to make the classification task easier, e.g.:
    * For classification of emails, you may remove signatures (or other decorators) to let the classifier focus on the content
    * For texts with domain-specific entities, you may normalize each entity into some canonical form that conveys its meaning


## Contact
If you have any questions, comments or suggestions - please reach out to [Oded Avraham](mailto:odeda@ai21.com) 👋🏼

For bug reports and feature requests - please visit our [GitHub page](https://github.com/AI21Labs/salt)
