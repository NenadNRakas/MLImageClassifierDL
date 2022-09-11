using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using static Microsoft.ML.DataOperationsCatalog;
using Microsoft.ML.Vision;

namespace MLImageClassifierDL
{
    internal class Program
    {
        static void Main(string[] args)
        {
            var projectDirectory = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../"));
            var workspaceRelativePath = Path.Combine(projectDirectory, "workspace");
            var assetsRelativePath = Path.Combine(projectDirectory, "assets");
            // Initialize            
            MLContext mlContext = new MLContext();
            // Get the list
            IEnumerable<ImageData> images = LoadImagesFromDirectory(folder: assetsRelativePath, useFolderNameAsLabel: true);
            // Load the images
            IDataView imageData = mlContext.Data.LoadFromEnumerable(images);
            // Shuffle the data
            IDataView shuffledData = mlContext.Data.ShuffleRows(imageData);
            // Numerical format label preprocess
            var preprocessingPipeline = mlContext.Transforms.Conversion.MapValueToKey
                (inputColumnName: "Label", outputColumnName: "LabelAsKey")
                .Append(mlContext.Transforms.LoadRawImageBytes
                (outputColumnName: "Image", imageFolder: assetsRelativePath, inputColumnName: "ImagePath"));
            // Apply the data to the pipeline
            IDataView preProcessedData = preprocessingPipeline.Fit(shuffledData).Transform(shuffledData);
            // Train, validate, test data
            TrainTestData trainSplit = mlContext.Data.TrainTestSplit(testFraction: 0.3, data: preProcessedData);
            TrainTestData validationTestSplit = mlContext.Data.TrainTestSplit(trainSplit.TestSet);
            // Assign partitions
            IDataView trainSet = trainSplit.TrainSet;
            IDataView validationSet = validationTestSplit.TrainSet;
            IDataView testSet = validationTestSplit.TestSet;
            // New training variable set
            var classifierOptions = new ImageClassificationTrainer.Options()
            {
                FeatureColumnName = "Image",
                LabelColumnName = "LabelAsKey",
                ValidationSet = validationSet,
                Arch = ImageClassificationTrainer.Architecture.ResnetV250,
                MetricsCallback = (metrics) => Console.WriteLine(metrics),
                TestOnTrainSet = false,
                ReuseTrainSetBottleneckCachedValues = false,
                ReuseValidationSetBottleneckCachedValues = false, WorkspacePath = workspaceRelativePath
            };
            // Define the pipeline
            var trainingPipeline = mlContext.MulticlassClassification.Trainers.ImageClassification(classifierOptions)
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
            // Train the model
            ITransformer trainedModel = trainingPipeline.Fit(trainSet);
            // Test
            ClassifySingleImage(mlContext, testSet, trainedModel);
            ClassifyImages(mlContext, testSet, trainedModel);
            Console.WriteLine("");
            Console.WriteLine("===========================Image Classification Complete=============================");
            Console.WriteLine("");
            Console.ReadKey();
        }
        // Single image classification
        public static void ClassifySingleImage(MLContext mlContext, IDataView data, ITransformer trainedModel)
        {
            // Pass in the data
            PredictionEngine<ModelInput, ModelOutput> predictionEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(trainedModel);
            // Convert the data
            ModelInput image = mlContext.Data.CreateEnumerable<ModelInput>(data, reuseRowObject: true).First();
            // Classify the image
            ModelOutput prediction = predictionEngine.Predict(image);
            // Display the output
            Console.WriteLine("");
            Console.WriteLine("===========================Single Image Classification===============================");
            Console.WriteLine("");
            OutputPrediction(prediction);
        }
        // MulticastDelegate image classification
        public static void ClassifyImages(MLContext mlContext, IDataView data, ITransformer trainedModel)
        {
            // Create the pediction data
            IDataView predictionData = trainedModel.Transform(data);
            // Create iterable data
            IEnumerable<ModelOutput> predictions = mlContext.Data.CreateEnumerable<ModelOutput>(predictionData, reuseRowObject: true).Take(10);
            // Iterate and output original and predicted labels
            Console.WriteLine("");
            Console.WriteLine("===========================Multiple Image Classification=============================");
            Console.WriteLine("");
            foreach (var prediction in predictions)
            {
                OutputPrediction(prediction);
            }
        }
        // Display the result
        private static void OutputPrediction(ModelOutput prediction)
        {
            string? imageName = Path.GetFileName(prediction.ImagePath);
            Console.WriteLine("---------------------------Image Classification Began--------------------------------");
            Console.WriteLine($"Image: {imageName} | Actual Location: {prediction.Label} | Probable State: {prediction.PredictedLabel}");
            Console.WriteLine("---------------------------Image Classification Ended--------------------------------");
        }
        // Data utilitty loading method
        public static IEnumerable<ImageData> LoadImagesFromDirectory(string folder, bool useFolderNameAsLabel = true)
        {
            var files = Directory.GetFiles(folder, "*", searchOption: SearchOption.AllDirectories);
            foreach (var file in files)
            {
                if ((Path.GetExtension(file) != ".jpg") && (Path.GetExtension(file) != ".png"))
                    continue;
                // Get the file label
                string? label = Path.GetFileName(file);
                if (useFolderNameAsLabel)
                    if (Directory.GetParent(file) == null) { label = Path.GetDirectoryName(file); }
                    else { label = Directory.GetParent(file).Name; } //label = Path.GetDirectoryName(file);
                else
                {
                    for (int index = 0; index < label.Length; index++)
                    {
                        if (!char.IsLetter(label[index]))
                        {
                            label = label.Substring(0, index);
                            break;
                        }
                    }
                }
                // Create a new instance
                yield return new ImageData()
                {
                    ImagePath = file,
                    Label = label
                };
            }
        }
    }
    public class ImageData
    {
        public string? ImagePath { get; set; }
        public string? Label { get; set; }
    }
    class ModelInput
    {
        public byte[]? Image { get; set; }

        public UInt32 LabelAsKey { get; set; }

        public string? ImagePath { get; set; }

        public string? Label { get; set; }
    }
    class ModelOutput
    {
        public string? ImagePath { get; set; }
        public string? Label { get; set; }
        public string? PredictedLabel { get; set; }
    }
}