# Drug Interaction Warning Application

This application predicts negative interactions between drugs using a machine learning model trained with ML.NET. Given the name of a drug, the application will list all drugs that have negative interactions with it, along with the type of interaction.

## Project Structure

DrugInteractionWarningApp/
├── .vscode/                          # Visual Studio Code configuration files

│   ├── settings.json                 # Project-specific settings

│   └── launch.json                   # Launch configurations for debugging

├── bin/                              # Compiled binaries and executables

│   └── Debug/                        # Debug configuration output

│       └── net6.0/                   # Output for .NET 6.0

├── data/                             # Directory containing datasets

│   └── DDI_data.csv                  # The dataset used for training the model

├── obj/                              # Object files and intermediate build artifacts

│   ├── Debug/                        # Debug configuration intermediates

│   └── project.assets.json           # Project assets file

├── .gitignore                        # Specifies intentionally untracked files to ignore

├── DrugInteractionWarningApp.csproj  # .NET project file with build configurations and dependencies

├── Program.cs                        # Main application code file

└── drug_interaction_model.zip        # Trained ML.NET model saved as a zip file

## Prerequisites

- [.NET SDK 6.0](https://dotnet.microsoft.com/download/dotnet/6.0) or higher

## How to Run

1. **Clone the repository**:
   Clone the project from GitHub to your local machine.

   ```bash
   git clone --branch master https://github.com/gemmap520/DrugInteractionWarningApp
   cd DrugInteractionWarningApp

2. **Build the project**:
   Use the .NET CLI to restore dependencies and build the project.

   ```bash
   dotnet build

3. **Run the application**:
   Execute the application to start the interactive console.

   ```bash
   dotnet run

4. **Enter a drug name**:
   After running the application, you will be prompted to enter the name of a drug. The application will then display a list of drugs that have negative interactions with the entered drug, along with the type of interaction.

## Dataset

The dataset `DDI_data.csv` is used for training and contains the following columns:

- **drug1_id**: ID of the first drug involved in the interaction
- **drug2_id**: ID of the second drug involved in the interaction
- **drug1_name**: Name of the first drug
- **drug2_name**: Name of the second drug
- **interaction_type**: Description of the type of interaction between the two drugs

## Code Overview

### `Program.cs`

This file contains the core logic of the application, including:

- **Loading Data**: Reads the `DDI_data.csv` file and loads it into an ML.NET `IDataView`.
- **Model Training**: Uses ML.NET to build and train a machine learning model for predicting drug interactions.
- **Model Evaluation**: Evaluates the model's accuracy using metrics such as MicroAccuracy and MacroAccuracy.
- **Prediction**: Once trained, the model predicts interactions based on user input.
- **Interactive Console**: Provides a console interface for users to input a drug name and receive a list of drugs that interact with it.

### How It Works

1. **Data Preparation**: The data is loaded and preprocessed using ML.NET's data transformations.
2. **Model Building**: The application uses a pipeline that includes text featurization and a multiclass classification trainer.
3. **Model Training**: The model is trained using the `SDCA Maximum Entropy` trainer, which is well-suited for multiclass classification tasks.
4. **Model Storage**: The trained model is saved to a zip file (`drug_interaction_model.zip`) for later use.
5. **Prediction Engine**: The application loads the model and uses a prediction engine to respond to user queries about drug interactions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
