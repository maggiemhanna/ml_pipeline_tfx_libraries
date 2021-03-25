# Production ML

Implementing and training an ML model with predictive performance on an offline holdout dataset, given relevant training data for our use case showed promising results. However, the real challenge isn't building an ML model, the challenge is building an integrated ML system and to continuously operate it in production.  Only a small fraction of a real-world ML system is composed of the ML code, the rest of the system is composed of configuration, automation, data collection, feature engineering, testing and debugging, resource management, model analysis, process and metadata management, serving infrastructure, and monitoring.

An ML system is a software system, so similar practices apply to help guarantee that we can reliably build and operate ML systems at scale. However, ML systems differ from other software systems in many ways:

* Development: ML is experimental in nature where different features, algorithms, modeling techniques, and parameter configurations are tried to find what works best for our problem. The challenge is tracking what worked and what didn’t and maintaining reproducibility while maximizing code reusability.
* Testing: In addition to typical unit and integration tests done in other software systems, ML systems need data validation, trained model quality evaluation, and model validation.
* Deployment: In ML systems, deployment isn't as simple as deploying an offline-trained ML model as a prediction service. ML systems require us to deploy a multi-step pipeline to automatically retrain and deploy the model. 
* Production: ML models can decay in more ways than conventional software systems as they can suffer from performance degradation due to constantly evolving data profiles. We need to track summary statistics of our data and monitor the online performance of our model to detect or rollback when values deviate from our expectations.

# ML Components

After defining the business use case and establishing the success criteria, the process of delivering an ML model involves the following components:
* **Data collection**

    * **Data ingestion**: This step includes selecting and merging the relevant data from various data sources (diagnostic database, warranty database…) for the ML task. Our initial data sources are stored on BigQuery. We prepare and clean our data by submitting a sequence of BigQuery and Spark jobs. With GCP Dataproc service (a fully managed service for running Apache Spark clusters) and BigQuery, we can profit from running fully distributed and scalable data extraction tasks in a simpler, more cost-efficient way without having to worry about the underlying infrastructure. The outputs of this step are data files in the parquet format on Google Storage, as well as a corresponding BigQuery table for the purpose of exploration.

    * **Exploratory data analysis**: In this step, we understand the available data for building the ML model in order to identify the data preparation and feature engineering techniques that might be interesting for the model. This step is also crucial at assessing data quality, such as missing values and outliers. This can be done in a notebook or exporting some interesting statistics and data characteristics to BigQuery.

* **Modeling**

    * **Data preparation / Feature Engineering**: The data is then prepared for the ML task. This involves data cleaning and filtering, data transformations, and feature engineering. In addition, this component prepares the trainer's metadata (the dictionaries needed for categorical variables encoding and features normalizatiion for example in the training step). These are called transformation artifacts; they help with constructing the model inputs. 
    * **Data validation**: In this step, we use TensorFlow Data Validation (TFDV); a library for exploring and validating machine learning data; to generate statistics about our datasets and validate the data against the expected (raw) data schema. The data schema is created during the development phase and is updated each time a new model is deployed. This step generates statistics about our training, validation, and test sets, and detects anomalies related to both data distribution and schema skews. It also helps to detect data drift by looking at a series of data. The outputs of this step are the anomalies and a decision on whether to execute downstream steps or not.
    This step is required before model training to decide whether we should retrain the model or stop the execution of the pipeline if the following was identified:
        * Data schema skews: These skews are considered anomalies in the input data, which means that the downstream steps, such as model training, receives data that doesn't comply with the expected schema. In this case, the data science team should investigate.
        * Data values skews: These skews are significant changes in the statistical properties of data, which means that data patterns are changing, and we need to trigger a retraining of the model to capture these changes.

    * **Model training**: This step involves training and hyperparameter tuning of the Machine Learning model. 
        * To implement and train our NN model, we use the TensorFlow tf.estimator API with the transformed data. Using TensorFlow estimators allows for several advantages. It allows us to focus on optimizing the ML algorithm while giving less attention to the boilerplate code that repeats itself every time. In addition to the standard models it provides, it lets us wrap our custom model that we build from layers using the TF layers (to implement our multi-label neural network architecture), TF losses (Macro F-score) and so on. Estimators are designed to work with tf.dataset API that handles out of memory data sets. It also provides checkpoints that allow us to pause and resume training when needed. Estimators also enable us to monitor the progress by surfacing our defined key metrics during training and visualizing them in Tensorboard. It can be distributed using the ParameterServer Strategy over a cluster to make training faster. It abstracts away the details of distributed execution for training and evaluation, while also supporting consistent behavior across local/non-distributed and distributed configurations.
        * GCP’s AI Platform provides the cluster configuration for running distributed TF training and provides a job management interface so that we don't need to manage the underlying infrastructure ourselves. Graphics Processing Units (GPUs) and Tensor Processing Units (TPUs) can also be used to accelerate machine-learning workloads. In addition, a scalable, Bayesian optimization-based service for a hyperparameter tuning is also available with AI platform, an approach that can achieve better performance while requiring fewer iterations than random search.   
        
        The output of this step is an EvalSavedModel that is used for evaluation, and another SavedModel that is used for online serving of the model for prediction. A SavedModel contains a complete TensorFlow program, including weights and computation. It does not require the original model building code to run, which makes it useful for sharing or deploying.
        
    * **Model evaluation and validation**: When the model is exported after the training step, it's evaluated on the test dataset to assess the model quality by using TensorFlow Model Analysis (TFMA), a library for evaluating TensorFlow models. It allows us to evaluate our models in a distributed manner, using the same metrics defined in our trainer. These metrics can be computed over different slices/segments (countries, engine types, symptoms, error codes, …) of data and visualized. We also track model performance over time so that we can be aware of and react to changes. This evaluation helps guarantee that the model is promoted for serving only if it satisfies the quality criteria. The criteria include improved performance compared to previous models and fair performance on various data subsets. The output of this step is a set of performance metrics (our evaluation metric F1score for all parts/operations) and a decision on whether to promote the model to production.

    * **Model serving**: The validated model is deployed to serve predictions. AI Platform provides a great infrastructure for this. It offers a scalable serverless rest API for real-time predictions and a batch service for the less latency-sensitive predictions.

* **Model monitoring**: The model predictive performance is monitored to potentially invoke a new iteration in the ML process.

# ML pipeline automation

The listed steps can be completed manually or can be completed by an automatic pipeline. A manual process is common in many businesses that are beginning to apply ML to their use cases. It might be sufficient when models are rarely changed or trained. In practice, models often break when they are deployed in the real world. The models fail to adapt to changes in the dynamics of the environment, or changes in the data that describes the environment. 

Opting for the manual process at the beginning, every step was manual, including data preparation, model training, and validation. It required manual execution of each step, and manual transition from one step to another. This process was driven by experimental code that is written and executed in notebooks interactively, until a workable model is produced.

A manual process was considered to be dangerous as it creates a disconnection between ML and operations. It separated data scientists who create the model and engineers who serve the model as a prediction service. The data scientists handed over a trained model as an artifact in a storage location to the engineering team to deploy on their API infrastructure. This process can definitely lead to training-serving skew.

The goal was then to perform continuous training of the model by automating the ML pipeline; which allows us to achieve continuous delivery of model prediction service. 

For this reason, we need an orchestrator in order to connect these different components of the system together. The orchestrator runs the pipeline in a sequence, and automatically moves from one step to another based on the defined conditions. For example, a defined condition might be executing the model serving step after the model evaluation step if the evaluation metrics meet predefined thresholds. Orchestrating the ML pipeline is useful in both the development and production phases:
* During the development phase, orchestration helps the data scientists to run the ML experiment, instead of manually executing each step which leads to rapid iteration of experiments and better readiness to move the whole pipeline to production.
* During the production phase, orchestration helps automate the execution of the ML pipeline on a schedule or certain triggering conditions (on demand, on a schedule, on availability of new training data, on model performance degradation, on significant changes in the data distributions…). An ML pipeline in production continuously trains and delivers prediction services to new models on fresh data automatically. With the manual process, we used to deploy a trained model as a prediction service to production. With the automated pipeline, we deploy a whole training pipeline.
* Experimental-operational symmetry: The pipeline implementation that is used in the development or experiment environment is used in the integration and production environment.

Kubeflow is an open source Kubernetes framework for developing and running portable ML workloads. Kubeflow Pipelines is a Kubeflow service that lets us compose, orchestrate, and automate ML systems, where each component/step of the system can run on Kubeflow, Google Cloud, or other cloud platforms. 

In our project, we opted to using 2 pipelines, one for data collection and another for ML modeling. The following constitutes our Kubeflow pipeline for ML modeling:
* A set of containerized ML tasks, or components for data preparation, data validation, model training, model validation, and model serving. A pipeline component is self-contained code that is packaged as a Docker image. A component takes input arguments, produces output files, and performs one step in the pipeline.
* A specification of the sequence of the ML tasks, defined through a Python domain-specific language (DSL). The topology of the workflow is implicitly defined by connecting the outputs of an upstream step to the inputs of a downstream step. A step in the pipeline definition invokes a component in the pipeline. 
* A set of pipeline input parameters (car model, TensorFlow version, …) whose values are passed to the components of the pipeline, including the criteria for filtering data and where to store the artifacts that the pipeline produces.


# CI/CD Automation

For a rapid and reliable update of the pipelines in production, we need a robust automated CI/CD system. This automated CI/CD system allows us to rapidly explore new ideas around feature engineering, model architecture, and hyperparameters. We can implement these ideas and automatically build, test, and deploy the new pipeline components to the target environment.

The CI/CD pipeline consists of the following stages:

* **Development and experimentation**: We iteratively try out new ML algorithms and new modeling where the experiment steps are orchestrated. The output of this stage is the source code of the ML pipeline steps that are then pushed to a source repository.

* **Pipeline continuous integration**: In this configuration, the pipeline and its components (creation and transmission of docker images, updating of component specifications, compilation of the kubeflow pipeline ...) are built, tested and packaged when the new code is validated or pushed to the deposit source code. The outputs of this stage are pipeline components (packages, executables, and artifacts) to be deployed in a later stage.

* **Pipeline continuous delivery**: At this level, our system continuously provides new pipeline implementations to the target environment which in turn provides predictive services for the newly formed model. The result of this step is a pipeline deployed with the new implementation of the model.

* **Automated triggering and Model continuous delivery**: The pipeline is automatically executed in production based on a schedule or in response to a trigger. We serve the trained model as a prediction service for the predictions. The output of this stage is a deployed model prediction service.

* **Monitoring**: We collect statistics on the model performance based on live data. The output of this stage is a trigger to execute the pipeline or to execute a new experiment cycle.
