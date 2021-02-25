# ML pipeline illustration with TFx libraries on Google Cloud Platform 

## General Description

This repo contains the implementation of a machine learning continouos training pipeline that orchestrates the continuous training and deployment of a neural network for bike sharing prediction model with TFx libraries on Google Cloud Platform.

The model uses the bike sharing demand data from kaggle as a use case for implementation: https://www.kaggle.com/c/bike-sharing-demand/data
The raw data from kaggle can be found 
```
doc/bike-sharing-demand
```

For educational purpose, some of the numeric variables (actually categorical) in the dataset have been changed to string.
This allow us to better interpret these variables (especially for TFMA slices analysis). The data is also copied to a gcp bucket.

You can check how data have been prepared in `doc/0_preparing_data`.

The train, val and test splits of the prepared data can be found at `doc/bike-sharing-data`, as well as the gs bucket `gs://bike-sharing-data`.


## Pipeline Components

The training ML pipeline has the following 5 components and is mainly orchestrated with Kubeflow Pipelines and run on Google Cloud AI Platform Infrastructure.

![Web App](img/pipeline_components.png)

We choose a functional decomposition of pipeline components to represent the main ML tasks.  
Each component is built as a docker container which ensures its isolation, autonomy and maintainability.

You can manually build a new version of each component by running the following command:
```
$ cd [containers/component_folder]
$ bash build.sh [project_id] [tag]
```

You can also find a notebook version of each of the components along side the python packages required to run the notebook in the `doc` directory.

This will create a new image of the component and push it to Google Container Registry.  
You can then provide the gcr image for each component and execute a Jupyter Notebook to compile the pipeline and submit it for execution with Kubeflow Pipelines.  
Example of Jupyter Notebooks for pipeline compilation may be found in `pipeline/bike_sharing_pipeline`.  

## Pipeline Parameters

### Global parameters
> * **pipeline_version:** Version of pipeline code being used which mainly contributes to name the model version
> * **region**: region used for underlying gcp infrastructure
> * **project:** GCP project Id where the pipelie will be executed
> * **bucket:** GCS bucket to store ML workflow data
> * **bucket_staging:** Bucket used for staging logs
> * **raw_data_path::** GCS path where prepared data is ingested
> * **tfversion:** Version of tensorflow package to use (e.g. 1.14)
> * **runner_validation:** Runner used for the data validation component (local or Dataflow)
> * **runner_transform:** Runner used for the data transform component (local or Dataflow)
> * **runner_training:** Runner used for the model training component (local or AIplatform)
> * **runner_analysis:** Runner used for the model analysis component (local or Dataflow)
> * **runner_deployment:** Runner used for the model deployment component (local or AIplatform)


### GCR docker images
> * **sin_validation:** Docker image on GCR of validation component
> * **sin_transform:** Docker image on GCR of transform component
> * **sin_hypertune:** Docker image on GCR of model training component
> * **sin_analysis:** Docker image on GCR of analysis component
> * **sin_deploy:** Docker image on GCR of deploy component

## Model Versioning
The pipeline when deployed is responsible of creating new ML models by refreshing on new coming data.  
We need 3 main information to define a unique model version deployed as inference endpoint on AI Platform:
> * The pipeline version that has been executed (e.g. v1.1)
> * The version of the dataset used for training and tuning the model represented by the date and time of preprocessing (e.g. 201231_125959)
> * The version of hyperparameter tuning represented by the date and time of searching for best model structure (e.g. 201231_125959

Example of model version: v0_3__200429_083501__200429_084356

# Next Steps 

* write pipeline `pipeline/workflow.py` script
* CI/CD pipeline with Cloud Build

# More Info

For more information about the ML production pipeline, please refer to the ProductionML.md