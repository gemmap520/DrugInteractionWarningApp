using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.IO;

public class DrugInteraction
{
    [LoadColumn(0)]
    public string Drug1ID { get; set; } = string.Empty;

    [LoadColumn(1)]
    public string Drug2ID { get; set; } = string.Empty;

    [LoadColumn(2)]
    public string Drug1Name { get; set; } = string.Empty;

    [LoadColumn(3)]
    public string Drug2Name { get; set; } = string.Empty;

    [LoadColumn(4)]
    public string InteractionType { get; set; } = string.Empty;
}

public class DrugPrediction
{
    [ColumnName("PredictedLabel")]
    public string? PredictedInteraction { get; set; }
}

public class Program
{
    static void Main(string[] args)
    {
        var mlContext = new MLContext();
        string dataPath = Path.Combine(Environment.CurrentDirectory, "data", "DDI_data.csv");

        var dataView = mlContext.Data.LoadFromTextFile<DrugInteraction>(dataPath, separatorChar: ',', hasHeader: true);

        var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", nameof(DrugInteraction.InteractionType))
            .Append(mlContext.Transforms.Text.FeaturizeText("Drug1Features", nameof(DrugInteraction.Drug1Name)))
            .Append(mlContext.Transforms.Text.FeaturizeText("Drug2Features", nameof(DrugInteraction.Drug2Name)))
            .Append(mlContext.Transforms.Concatenate("Features", "Drug1Features", "Drug2Features"))
            .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy())
            .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

        var model = pipeline.Fit(dataView);

        var testData = mlContext.Data.LoadFromTextFile<DrugInteraction>(dataPath, separatorChar: ',', hasHeader: true);
        var predictions = model.Transform(testData);
        var metrics = mlContext.MulticlassClassification.Evaluate(predictions);

        Console.WriteLine($"MicroAccuracy: {metrics.MicroAccuracy}, MacroAccuracy: {metrics.MacroAccuracy}");

        mlContext.Model.Save(model, dataView.Schema, "drug_interaction_model.zip");

        ITransformer loadedModel;
        DataViewSchema modelSchema;
        using (var fileStream = new FileStream("drug_interaction_model.zip", FileMode.Open, FileAccess.Read, FileShare.Read))
        {
            loadedModel = mlContext.Model.Load(fileStream, out modelSchema);
        }

        Console.WriteLine("Enter a drug name:");
        string drugName = Console.ReadLine();

        var predictionEngine = mlContext.Model.CreatePredictionEngine<DrugInteraction, DrugPrediction>(loadedModel);

        // Load the entire dataset into a list for easy querying
        var drugData = mlContext.Data.CreateEnumerable<DrugInteraction>(dataView, reuseRowObject: false);

        // Find all drugs that interact with the entered drug name
        var interactingDrugs = new List<DrugInteraction>();

        foreach (var drugInteraction in drugData)
        {
            if (drugInteraction.Drug1Name.Equals(drugName, StringComparison.OrdinalIgnoreCase) ||
                drugInteraction.Drug2Name.Equals(drugName, StringComparison.OrdinalIgnoreCase))
            {
                interactingDrugs.Add(drugInteraction);
            }
        }

        // Print all interacting drugs and their interaction types
        if (interactingDrugs.Count > 0)
        {
            Console.WriteLine($"Drugs that interact with {drugName}:");
            foreach (var interaction in interactingDrugs)
            {
                var interactingDrugName = interaction.Drug1Name.Equals(drugName, StringComparison.OrdinalIgnoreCase) ? interaction.Drug2Name : interaction.Drug1Name;
                Console.WriteLine($"- {interactingDrugName}: {interaction.InteractionType}");
            }
        }
        else
        {
            Console.WriteLine($"No interactions found for {drugName}.");
        }
    }
}

// using System;
// using Microsoft.ML;
// using Microsoft.ML.Data;
// using System.IO;

// public class DrugInteraction
// {
//     [LoadColumn(0)]
//     public string Drug1ID { get; set; } = string.Empty;

//     [LoadColumn(1)]
//     public string Drug2ID { get; set; } = string.Empty;

//     [LoadColumn(2)]
//     public string Drug1Name { get; set; } = string.Empty;

//     [LoadColumn(3)]
//     public string Drug2Name { get; set; } = string.Empty;

//     [LoadColumn(4)]
//     public string InteractionType { get; set; } = string.Empty;
// }

// public class DrugPrediction
// {
//     [ColumnName("PredictedLabel")]
//     public string? PredictedInteraction { get; set; }
// }

// public class Program
// {
//     static void Main(string[] args)
//     {
//         var mlContext = new MLContext();
//         string dataPath = Path.Combine(Environment.CurrentDirectory, "data", "DDI_data.csv");

//         var dataView = mlContext.Data.LoadFromTextFile<DrugInteraction>(dataPath, separatorChar: ',', hasHeader: true);

//         var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", nameof(DrugInteraction.InteractionType))
//             .Append(mlContext.Transforms.Text.FeaturizeText("Drug1Features", nameof(DrugInteraction.Drug1Name)))
//             .Append(mlContext.Transforms.Text.FeaturizeText("Drug2Features", nameof(DrugInteraction.Drug2Name)))
//             .Append(mlContext.Transforms.Concatenate("Features", "Drug1Features", "Drug2Features"))
//             .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy())
//             .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

//         var model = pipeline.Fit(dataView);

//         var testData = mlContext.Data.LoadFromTextFile<DrugInteraction>(dataPath, separatorChar: ',', hasHeader: true);
//         var predictions = model.Transform(testData);
//         var metrics = mlContext.MulticlassClassification.Evaluate(predictions);

//         Console.WriteLine($"MicroAccuracy: {metrics.MicroAccuracy}, MacroAccuracy: {metrics.MacroAccuracy}");

//         mlContext.Model.Save(model, dataView.Schema, "drug_interaction_model.zip");

//         ITransformer loadedModel;
//         DataViewSchema modelSchema;
//         using (var fileStream = new FileStream("drug_interaction_model.zip", FileMode.Open, FileAccess.Read, FileShare.Read))
//         {
//             loadedModel = mlContext.Model.Load(fileStream, out modelSchema);
//         }

//         Console.WriteLine("Enter the first drug name:");
//         string drug1Name = Console.ReadLine();

//         Console.WriteLine("Enter the second drug name:");
//         string drug2Name = Console.ReadLine();

//         var predictionEngine = mlContext.Model.CreatePredictionEngine<DrugInteraction, DrugPrediction>(loadedModel);
//         var prediction = predictionEngine.Predict(new DrugInteraction { Drug1Name = drug1Name, Drug2Name = drug2Name });

//         Console.WriteLine($"Predicted Interaction Type: {prediction.PredictedInteraction}");
//     }
// }
