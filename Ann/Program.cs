using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using BackPropagationNetwork;
using BackPropagationNetwork.DebugLogger;

namespace Ann
{
    internal class Program
    {
        private static readonly NeuralNetwork NeuralNetwork = new NeuralNetwork(new ConsoleDebugLogger(), 0);
        private static readonly double[][] AllData = new double[150][];

        private const string SaveFileName = "Weights.txt";

        public static void Main(string[] args)
        {
            Program.SetData();

            for (;;)
            {
                Console.WriteLine("1. Train");
                Console.WriteLine("2. Save Weights");
                Console.WriteLine("3. Load Weights");
                Console.WriteLine("4. Test Accuracy with full set.");
                Console.WriteLine("5. Exit");
                Console.WriteLine();

                ConsoleKeyInfo keyInfo = Console.ReadKey();
                Console.WriteLine();

                var exit = false;

                switch (keyInfo.KeyChar)
                {
                    case '1':
                        Program.Train();
                        break;

                    case '2':
                        Program.SaveWeights();
                        break;

                    case '3':
                        Program.LoadWeights();
                        break;

                    case '4':
                        Program.TestAccuracy();
                        break;

                    case '5':
                        exit = true;
                        break;

                    default:
                        Console.WriteLine("Unknown input");
                        break;
                }

                if (exit)
                    break;

                Console.WriteLine();
            }
        }

        private static void TestAccuracy()
        {
            Data data = Program.GetDataObject(Program.AllData);
            double accuracy = Program.NeuralNetwork.GetAccuracy(data.Elements, data.Targets);

            Console.WriteLine($"Accuracy for all data: {accuracy:F4}");
        }

        private static void LoadWeights()
        {
            var weights = new List<double>();

            using (var reader = new StreamReader(Program.SaveFileName))
            {
                string input;
                while ((input = reader.ReadLine()) != null)
                weights.Add(Convert.ToDouble(input));
            }

            Program.NeuralNetwork.SetWeightsAndBiases(weights.ToArray());
        }

        private static void SaveWeights()
        {
            double[] weights = Program.NeuralNetwork.GetWeightsAndBiases();            

            using (var writer = new StreamWriter(Program.SaveFileName))
            {
                weights.ToList().ForEach(w => writer.WriteLine(w.ToString("G17")));
            }
        }

        private static void SetData()
        {
            #region Data

            Program.AllData[0] = new[] {5.1, 3.5, 1.4, 0.2, 0, 0, 1};
                // sepal length, sepal width, petal length, petal width -> 
            Program.AllData[1] = new[] {4.9, 3.0, 1.4, 0.2, 0, 0, 1};
                // Iris setosa = 0 0 1, Iris versicolor = 0 1 0, Iris virginica = 1 0 0
            Program.AllData[2] = new[] {4.7, 3.2, 1.3, 0.2, 0, 0, 1};
            Program.AllData[3] = new[] {4.6, 3.1, 1.5, 0.2, 0, 0, 1};
            Program.AllData[4] = new[] {5.0, 3.6, 1.4, 0.2, 0, 0, 1};
            Program.AllData[5] = new[] {5.4, 3.9, 1.7, 0.4, 0, 0, 1};
            Program.AllData[6] = new[] {4.6, 3.4, 1.4, 0.3, 0, 0, 1};
            Program.AllData[7] = new[] {5.0, 3.4, 1.5, 0.2, 0, 0, 1};
            Program.AllData[8] = new[] {4.4, 2.9, 1.4, 0.2, 0, 0, 1};
            Program.AllData[9] = new[] {4.9, 3.1, 1.5, 0.1, 0, 0, 1};

            Program.AllData[10] = new[] {5.4, 3.7, 1.5, 0.2, 0, 0, 1};
            Program.AllData[11] = new[] {4.8, 3.4, 1.6, 0.2, 0, 0, 1};
            Program.AllData[12] = new[] {4.8, 3.0, 1.4, 0.1, 0, 0, 1};
            Program.AllData[13] = new[] {4.3, 3.0, 1.1, 0.1, 0, 0, 1};
            Program.AllData[14] = new[] {5.8, 4.0, 1.2, 0.2, 0, 0, 1};
            Program.AllData[15] = new[] {5.7, 4.4, 1.5, 0.4, 0, 0, 1};
            Program.AllData[16] = new[] {5.4, 3.9, 1.3, 0.4, 0, 0, 1};
            Program.AllData[17] = new[] {5.1, 3.5, 1.4, 0.3, 0, 0, 1};
            Program.AllData[18] = new[] {5.7, 3.8, 1.7, 0.3, 0, 0, 1};
            Program.AllData[19] = new[] {5.1, 3.8, 1.5, 0.3, 0, 0, 1};

            Program.AllData[20] = new[] {5.4, 3.4, 1.7, 0.2, 0, 0, 1};
            Program.AllData[21] = new[] {5.1, 3.7, 1.5, 0.4, 0, 0, 1};
            Program.AllData[22] = new[] {4.6, 3.6, 1.0, 0.2, 0, 0, 1};
            Program.AllData[23] = new[] {5.1, 3.3, 1.7, 0.5, 0, 0, 1};
            Program.AllData[24] = new[] {4.8, 3.4, 1.9, 0.2, 0, 0, 1};
            Program.AllData[25] = new[] {5.0, 3.0, 1.6, 0.2, 0, 0, 1};
            Program.AllData[26] = new[] {5.0, 3.4, 1.6, 0.4, 0, 0, 1};
            Program.AllData[27] = new[] {5.2, 3.5, 1.5, 0.2, 0, 0, 1};
            Program.AllData[28] = new[] {5.2, 3.4, 1.4, 0.2, 0, 0, 1};
            Program.AllData[29] = new[] {4.7, 3.2, 1.6, 0.2, 0, 0, 1};

            Program.AllData[30] = new[] {4.8, 3.1, 1.6, 0.2, 0, 0, 1};
            Program.AllData[31] = new[] {5.4, 3.4, 1.5, 0.4, 0, 0, 1};
            Program.AllData[32] = new[] {5.2, 4.1, 1.5, 0.1, 0, 0, 1};
            Program.AllData[33] = new[] {5.5, 4.2, 1.4, 0.2, 0, 0, 1};
            Program.AllData[34] = new[] {4.9, 3.1, 1.5, 0.1, 0, 0, 1};
            Program.AllData[35] = new[] {5.0, 3.2, 1.2, 0.2, 0, 0, 1};
            Program.AllData[36] = new[] {5.5, 3.5, 1.3, 0.2, 0, 0, 1};
            Program.AllData[37] = new[] {4.9, 3.1, 1.5, 0.1, 0, 0, 1};
            Program.AllData[38] = new[] {4.4, 3.0, 1.3, 0.2, 0, 0, 1};
            Program.AllData[39] = new[] {5.1, 3.4, 1.5, 0.2, 0, 0, 1};

            Program.AllData[40] = new[] {5.0, 3.5, 1.3, 0.3, 0, 0, 1};
            Program.AllData[41] = new[] {4.5, 2.3, 1.3, 0.3, 0, 0, 1};
            Program.AllData[42] = new[] {4.4, 3.2, 1.3, 0.2, 0, 0, 1};
            Program.AllData[43] = new[] {5.0, 3.5, 1.6, 0.6, 0, 0, 1};
            Program.AllData[44] = new[] {5.1, 3.8, 1.9, 0.4, 0, 0, 1};
            Program.AllData[45] = new[] {4.8, 3.0, 1.4, 0.3, 0, 0, 1};
            Program.AllData[46] = new[] {5.1, 3.8, 1.6, 0.2, 0, 0, 1};
            Program.AllData[47] = new[] {4.6, 3.2, 1.4, 0.2, 0, 0, 1};
            Program.AllData[48] = new[] {5.3, 3.7, 1.5, 0.2, 0, 0, 1};
            Program.AllData[49] = new[] {5.0, 3.3, 1.4, 0.2, 0, 0, 1};

            Program.AllData[50] = new[] {7.0, 3.2, 4.7, 1.4, 0, 1, 0};
            Program.AllData[51] = new[] {6.4, 3.2, 4.5, 1.5, 0, 1, 0};
            Program.AllData[52] = new[] {6.9, 3.1, 4.9, 1.5, 0, 1, 0};
            Program.AllData[53] = new[] {5.5, 2.3, 4.0, 1.3, 0, 1, 0};
            Program.AllData[54] = new[] {6.5, 2.8, 4.6, 1.5, 0, 1, 0};
            Program.AllData[55] = new[] {5.7, 2.8, 4.5, 1.3, 0, 1, 0};
            Program.AllData[56] = new[] {6.3, 3.3, 4.7, 1.6, 0, 1, 0};
            Program.AllData[57] = new[] {4.9, 2.4, 3.3, 1.0, 0, 1, 0};
            Program.AllData[58] = new[] {6.6, 2.9, 4.6, 1.3, 0, 1, 0};
            Program.AllData[59] = new[] {5.2, 2.7, 3.9, 1.4, 0, 1, 0};

            Program.AllData[60] = new[] {5.0, 2.0, 3.5, 1.0, 0, 1, 0};
            Program.AllData[61] = new[] {5.9, 3.0, 4.2, 1.5, 0, 1, 0};
            Program.AllData[62] = new[] {6.0, 2.2, 4.0, 1.0, 0, 1, 0};
            Program.AllData[63] = new[] {6.1, 2.9, 4.7, 1.4, 0, 1, 0};
            Program.AllData[64] = new[] {5.6, 2.9, 3.6, 1.3, 0, 1, 0};
            Program.AllData[65] = new[] {6.7, 3.1, 4.4, 1.4, 0, 1, 0};
            Program.AllData[66] = new[] {5.6, 3.0, 4.5, 1.5, 0, 1, 0};
            Program.AllData[67] = new[] {5.8, 2.7, 4.1, 1.0, 0, 1, 0};
            Program.AllData[68] = new[] {6.2, 2.2, 4.5, 1.5, 0, 1, 0};
            Program.AllData[69] = new[] {5.6, 2.5, 3.9, 1.1, 0, 1, 0};

            Program.AllData[70] = new[] {5.9, 3.2, 4.8, 1.8, 0, 1, 0};
            Program.AllData[71] = new[] {6.1, 2.8, 4.0, 1.3, 0, 1, 0};
            Program.AllData[72] = new[] {6.3, 2.5, 4.9, 1.5, 0, 1, 0};
            Program.AllData[73] = new[] {6.1, 2.8, 4.7, 1.2, 0, 1, 0};
            Program.AllData[74] = new[] {6.4, 2.9, 4.3, 1.3, 0, 1, 0};
            Program.AllData[75] = new[] {6.6, 3.0, 4.4, 1.4, 0, 1, 0};
            Program.AllData[76] = new[] {6.8, 2.8, 4.8, 1.4, 0, 1, 0};
            Program.AllData[77] = new[] {6.7, 3.0, 5.0, 1.7, 0, 1, 0};
            Program.AllData[78] = new[] {6.0, 2.9, 4.5, 1.5, 0, 1, 0};
            Program.AllData[79] = new[] {5.7, 2.6, 3.5, 1.0, 0, 1, 0};

            Program.AllData[80] = new[] {5.5, 2.4, 3.8, 1.1, 0, 1, 0};
            Program.AllData[81] = new[] {5.5, 2.4, 3.7, 1.0, 0, 1, 0};
            Program.AllData[82] = new[] {5.8, 2.7, 3.9, 1.2, 0, 1, 0};
            Program.AllData[83] = new[] {6.0, 2.7, 5.1, 1.6, 0, 1, 0};
            Program.AllData[84] = new[] {5.4, 3.0, 4.5, 1.5, 0, 1, 0};
            Program.AllData[85] = new[] {6.0, 3.4, 4.5, 1.6, 0, 1, 0};
            Program.AllData[86] = new[] {6.7, 3.1, 4.7, 1.5, 0, 1, 0};
            Program.AllData[87] = new[] {6.3, 2.3, 4.4, 1.3, 0, 1, 0};
            Program.AllData[88] = new[] {5.6, 3.0, 4.1, 1.3, 0, 1, 0};
            Program.AllData[89] = new[] {5.5, 2.5, 4.0, 1.3, 0, 1, 0};

            Program.AllData[90] = new[] {5.5, 2.6, 4.4, 1.2, 0, 1, 0};
            Program.AllData[91] = new[] {6.1, 3.0, 4.6, 1.4, 0, 1, 0};
            Program.AllData[92] = new[] {5.8, 2.6, 4.0, 1.2, 0, 1, 0};
            Program.AllData[93] = new[] {5.0, 2.3, 3.3, 1.0, 0, 1, 0};
            Program.AllData[94] = new[] {5.6, 2.7, 4.2, 1.3, 0, 1, 0};
            Program.AllData[95] = new[] {5.7, 3.0, 4.2, 1.2, 0, 1, 0};
            Program.AllData[96] = new[] {5.7, 2.9, 4.2, 1.3, 0, 1, 0};
            Program.AllData[97] = new[] {6.2, 2.9, 4.3, 1.3, 0, 1, 0};
            Program.AllData[98] = new[] {5.1, 2.5, 3.0, 1.1, 0, 1, 0};
            Program.AllData[99] = new[] {5.7, 2.8, 4.1, 1.3, 0, 1, 0};

            Program.AllData[100] = new[] {6.3, 3.3, 6.0, 2.5, 1, 0, 0};
            Program.AllData[101] = new[] {5.8, 2.7, 5.1, 1.9, 1, 0, 0};
            Program.AllData[102] = new[] {7.1, 3.0, 5.9, 2.1, 1, 0, 0};
            Program.AllData[103] = new[] {6.3, 2.9, 5.6, 1.8, 1, 0, 0};
            Program.AllData[104] = new[] {6.5, 3.0, 5.8, 2.2, 1, 0, 0};
            Program.AllData[105] = new[] {7.6, 3.0, 6.6, 2.1, 1, 0, 0};
            Program.AllData[106] = new[] {4.9, 2.5, 4.5, 1.7, 1, 0, 0};
            Program.AllData[107] = new[] {7.3, 2.9, 6.3, 1.8, 1, 0, 0};
            Program.AllData[108] = new[] {6.7, 2.5, 5.8, 1.8, 1, 0, 0};
            Program.AllData[109] = new[] {7.2, 3.6, 6.1, 2.5, 1, 0, 0};

            Program.AllData[110] = new[] {6.5, 3.2, 5.1, 2.0, 1, 0, 0};
            Program.AllData[111] = new[] {6.4, 2.7, 5.3, 1.9, 1, 0, 0};
            Program.AllData[112] = new[] {6.8, 3.0, 5.5, 2.1, 1, 0, 0};
            Program.AllData[113] = new[] {5.7, 2.5, 5.0, 2.0, 1, 0, 0};
            Program.AllData[114] = new[] {5.8, 2.8, 5.1, 2.4, 1, 0, 0};
            Program.AllData[115] = new[] {6.4, 3.2, 5.3, 2.3, 1, 0, 0};
            Program.AllData[116] = new[] {6.5, 3.0, 5.5, 1.8, 1, 0, 0};
            Program.AllData[117] = new[] {7.7, 3.8, 6.7, 2.2, 1, 0, 0};
            Program.AllData[118] = new[] {7.7, 2.6, 6.9, 2.3, 1, 0, 0};
            Program.AllData[119] = new[] {6.0, 2.2, 5.0, 1.5, 1, 0, 0};

            Program.AllData[120] = new[] {6.9, 3.2, 5.7, 2.3, 1, 0, 0};
            Program.AllData[121] = new[] {5.6, 2.8, 4.9, 2.0, 1, 0, 0};
            Program.AllData[122] = new[] {7.7, 2.8, 6.7, 2.0, 1, 0, 0};
            Program.AllData[123] = new[] {6.3, 2.7, 4.9, 1.8, 1, 0, 0};
            Program.AllData[124] = new[] {6.7, 3.3, 5.7, 2.1, 1, 0, 0};
            Program.AllData[125] = new[] {7.2, 3.2, 6.0, 1.8, 1, 0, 0};
            Program.AllData[126] = new[] {6.2, 2.8, 4.8, 1.8, 1, 0, 0};
            Program.AllData[127] = new[] {6.1, 3.0, 4.9, 1.8, 1, 0, 0};
            Program.AllData[128] = new[] {6.4, 2.8, 5.6, 2.1, 1, 0, 0};
            Program.AllData[129] = new[] {7.2, 3.0, 5.8, 1.6, 1, 0, 0};

            Program.AllData[130] = new[] {7.4, 2.8, 6.1, 1.9, 1, 0, 0};
            Program.AllData[131] = new[] {7.9, 3.8, 6.4, 2.0, 1, 0, 0};
            Program.AllData[132] = new[] {6.4, 2.8, 5.6, 2.2, 1, 0, 0};
            Program.AllData[133] = new[] {6.3, 2.8, 5.1, 1.5, 1, 0, 0};
            Program.AllData[134] = new[] {6.1, 2.6, 5.6, 1.4, 1, 0, 0};
            Program.AllData[135] = new[] {7.7, 3.0, 6.1, 2.3, 1, 0, 0};
            Program.AllData[136] = new[] {6.3, 3.4, 5.6, 2.4, 1, 0, 0};
            Program.AllData[137] = new[] {6.4, 3.1, 5.5, 1.8, 1, 0, 0};
            Program.AllData[138] = new[] {6.0, 3.0, 4.8, 1.8, 1, 0, 0};
            Program.AllData[139] = new[] {6.9, 3.1, 5.4, 2.1, 1, 0, 0};

            Program.AllData[140] = new[] {6.7, 3.1, 5.6, 2.4, 1, 0, 0};
            Program.AllData[141] = new[] {6.9, 3.1, 5.1, 2.3, 1, 0, 0};
            Program.AllData[142] = new[] {5.8, 2.7, 5.1, 1.9, 1, 0, 0};
            Program.AllData[143] = new[] {6.8, 3.2, 5.9, 2.3, 1, 0, 0};
            Program.AllData[144] = new[] {6.7, 3.3, 5.7, 2.5, 1, 0, 0};
            Program.AllData[145] = new[] {6.7, 3.0, 5.2, 2.3, 1, 0, 0};
            Program.AllData[146] = new[] {6.3, 2.5, 5.0, 1.9, 1, 0, 0};
            Program.AllData[147] = new[] {6.5, 3.0, 5.2, 2.0, 1, 0, 0};
            Program.AllData[148] = new[] {6.2, 3.4, 5.4, 2.3, 1, 0, 0};
            Program.AllData[149] = new[] {5.9, 3.0, 5.1, 1.8, 1, 0, 0};
            
            #endregion Data
        }

        private static void Train()
        {
            Data data = Program.GetDataObject(Program.AllData);

            const int maxEpochs = 5000;
            const double learnRate = 0.05;
            const double momentum = 0.01;

            Program.NeuralNetwork.Train(data, maxEpochs, learnRate, momentum,
                data.NumberOfElementsPerSet + 3); // Magic number

            double trainingAccuracy = Program.NeuralNetwork.GetTrainingAccuracy();
            Console.WriteLine($"Training accuracy: {trainingAccuracy:F4}");

            double testAccuracy = Program.NeuralNetwork.GetTestAccuracy();
            Console.WriteLine($"Test accuracy {testAccuracy:F4}");
        }

        private static Data GetDataObject(double[][] allData)
        {
            const int inputRowLength = 4;
            const int targetArraySize = 3;

            var data = new Data((ulong) allData.Length, inputRowLength, targetArraySize);

            for (var i = 0; i < allData.Length; i++)
            {
                var input = new double[inputRowLength];

                Array.Copy(allData[i], input, inputRowLength);

                var target = 0;
                for (var j = 2; j >= 0; j--)
                    target += (int) Math.Pow(2, j)*(int) allData[i][j + inputRowLength];

                data.Add(input, target);
            }

            return data;
        }
    }
}