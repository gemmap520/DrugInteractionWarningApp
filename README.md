# Drug Interaction Warning Application

This application predicts negative interactions between drugs using a machine learning model trained with ML.NET. Given the name of a drug, the application will list all drugs that have negative interactions with it, along with the type of interaction.

## Project Structure
- **.vscode/**: Configuration files for Visual Studio Code.
- **bin/**: Output directory for compiled binaries.
- **data/**: Contains the `DDI_data.csv` file, which is the dataset used for training the model.
- **obj/**: Intermediate files and object files directory.
- **.gitignore**: Git ignore file specifying files and directories to be ignored by Git.
- **DrugInteractionWarningApp.csproj**: The project file containing build configurations and package references.
- **Program.cs**: The main application code.
- **drug_interaction_model.zip**: The trained ML.NET model.

## Prerequisites

- [.NET SDK](https://dotnet.microsoft.com/download) 6.0 or higher

## How to Run

1. **Clone the repository**:
   ```bash
   git clone <https://github.com/gemmap520/DrugInteractionWarningApp.git>
   cd DrugInteractionWarningApp

