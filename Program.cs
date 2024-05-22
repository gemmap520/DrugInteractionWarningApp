using System;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.IO;

public class Drug
{
    [LoadColumn(0)]
    public string DrugName { get; set; }

    [LoadColumn(1)]
    public string InteractionDrug { get; set; }

    [LoadColumn(2)]
    public string InteractionDescription { get; set; }
}

public class DrugPrediction : Drug
{
    public string? PredictedLabel { get; set; }
}

public class Program
{
    static void Main(string[] args)
    {
        var mlContext = new MLContext();
        string dataPath = Path.Combine(Environment.CurrentDirectory, "data", "drug_interactions.csv");

        var dataView = mlContext.Data.LoadFromTextFile<Drug>(dataPath, separatorChar: ',', hasHeader: true);

        var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", nameof(Drug.InteractionDrug))
            .Append(mlContext.Transforms.Text.FeaturizeText("Features", nameof(Drug.DrugName)))
            .Append(mlContext.Transforms.Concatenate("Features", "Features"))
            .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy())
            .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

        var model = pipeline.Fit(dataView);

        var testData = mlContext.Data.LoadFromTextFile<Drug>(dataPath, separatorChar: ',', hasHeader: true);
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

        var predictionEngine = mlContext.Model.CreatePredictionEngine<Drug, DrugPrediction>(loadedModel);
        var prediction = predictionEngine.Predict(new Drug { DrugName = drugName });

        Console.WriteLine($"Predicted Interaction Drug: {prediction.PredictedLabel}");
        Console.WriteLine($"Description: {prediction.InteractionDescription}");
    }
}
