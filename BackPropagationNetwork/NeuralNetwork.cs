using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using BackPropagationNetwork.DebugLogger;

namespace BackPropagationNetwork
{
    public class NeuralNetwork
    {
        private Random random;

        private int numberOfInputs;
        private int numberOfHiddenLayerNeurons;
        private int numberOfOutputLayerNeurons;

        private double[] inputs;

        private double[][] inputLayerTohiddenLayerWeights;
        private double[] hiddenLayerBiases;
        private double[] hiddenLayerOutputs;

        private double[][] hiddenLayerToOutputLayerWeights;
        private double[] outputLayerBiases;

        private double[] outputs;

        private double[] hiddenLayerGradians;
        private double[] outputLayerGradians;

        private double[][] inputLayerToHiddenLayer_PreviousWeightsDeltas;
        private double[] hiddenLayerPreviousBiasDeltas;
        private double[][] hiddenLayerToOutputLayer_PreviousWeightsDeltas;
        private double[] outputLayerPreviousBiasDeltas;

        private double[][] trainingData;
        private int[][] trainingTargets;

        private double[][] testData;
        private int[][] testTargets;

        private readonly IDebugLogger infoLogger;

        private readonly int? randomSeed;

        public NeuralNetwork(IDebugLogger infoLogger, int? randomSeed = null)
        {
            this.infoLogger = infoLogger;
            this.randomSeed = randomSeed;
            this.random = NeuralNetwork.GetRandomizer(this.randomSeed);
        }

        private static Random GetRandomizer(int? randomSeed)
        {
            return randomSeed == null ? new Random() : new Random(randomSeed.Value);
        }

        public void Train(Data data,
            int maxEprochs, double learnRate, double momentum, int numberOfHiddenLayerNeurons = -1)
        {
            this.numberOfInputs = data.NumberOfElementsPerSet;
            this.numberOfOutputLayerNeurons = data.NumberOfDiscreteTargetValues;

            if (numberOfHiddenLayerNeurons < this.numberOfInputs || numberOfHiddenLayerNeurons < this.numberOfOutputLayerNeurons)
            {
                this.numberOfHiddenLayerNeurons = Math.Max(this.numberOfInputs, this.numberOfOutputLayerNeurons) + 3;  // Magic number
            }
            else
            {
                this.numberOfHiddenLayerNeurons = numberOfHiddenLayerNeurons;
            }

            if (this.numberOfInputs != data.NumberOfElementsPerSet ||
                this.numberOfOutputLayerNeurons != data.NumberOfDiscreteTargetValues)
            {
                throw new NeuralNetworkException("Data shape does not match network initialization parameters.");
            }

            this.AllocateSpace();
            this.InitializeWeights();

            this.MakeTrainTestData(data);

            //this.Normalize();

            this.Train(maxEprochs, learnRate, momentum);
        }

        public double GetTrainingAccuracy()
        {
            return this.GetAccuracy(this.trainingData, this.trainingTargets);
        }

        public double GetTestAccuracy()
        {
            return this.GetAccuracy(this.testData, this.testTargets);
        }

        public double[] GetWeightsAndBiases()
        {
            var resultList = new List<double>();

            resultList.Add(this.numberOfInputs);
            resultList.Add(this.numberOfHiddenLayerNeurons);
            resultList.Add(this.numberOfOutputLayerNeurons);
            NeuralNetwork.AddArray(this.inputLayerTohiddenLayerWeights, resultList);
            NeuralNetwork.AddArray(this.hiddenLayerBiases, resultList);
            NeuralNetwork.AddArray(this.hiddenLayerToOutputLayerWeights, resultList);
            NeuralNetwork.AddArray(this.outputLayerBiases, resultList);
            return resultList.ToArray();
        }

        public void SetWeightsAndBiases(double[] input)
        {
            this.random = NeuralNetwork.GetRandomizer(this.randomSeed);

            int j = 0;

            this.numberOfInputs = (int)input[j++];
            this.numberOfHiddenLayerNeurons = (int)input[j++];
            this.numberOfOutputLayerNeurons = (int)input[j++];

            this.AllocateSpace();

            NeuralNetwork.SetArray(this.inputLayerTohiddenLayerWeights, input, ref j);
            NeuralNetwork.SetArray(this.hiddenLayerBiases, input, ref j);
            NeuralNetwork.SetArray(this.hiddenLayerToOutputLayerWeights, input, ref j);
            NeuralNetwork.SetArray(this.outputLayerBiases, input, ref j);
        }

        private static void AddArray(double[][] input, List<double> list)
        {
            input.ToList()
                .ForEach(innerArray => NeuralNetwork.AddArray(innerArray, list));
        }

        private static void AddArray(double[] input, List<double> list)
        {
            input.ToList().ForEach(list.Add);
        }

        private static void SetArray(double[][] array, double[] input, ref int pointer)
        {
            for (int i = 0; i < array.Length; i++)
            {
                for (int j = 0; j < array[0].Length; j++)
                {
                    array[i][j] = input[pointer++];
                }
            }
        }

        private static void SetArray(double[] array, double[] input, ref int pointer)
        {
            Array.Copy(input, pointer, array, 0, array.Length);
            pointer += array.Length;
        }

        private void Normalize()
        {
            NeuralNetwork.Normalize(this.trainingData);
            NeuralNetwork.Normalize(this.testData);
        }

        private static void Normalize(double[][] dataMatrix)
        {
            if (!dataMatrix.Any())
            {
                throw new NeuralNetworkException("Array to be normalized is zero-length");
            }

            int numberOfColumns = dataMatrix[0].Length;

            // normalize specified cols by computing (x - mean) / sd for each value
            for (int col = 0; col < numberOfColumns; col++)
            {
                double sum = 0.0;
                for (int i = 0; i < dataMatrix.Length; ++i)
                    sum += dataMatrix[i][col];
                double mean = sum / dataMatrix.Length;
                sum = 0.0;
                for (int i = 0; i < dataMatrix.Length; ++i)
                    sum += (dataMatrix[i][col] - mean) * (dataMatrix[i][col] - mean);
                double sd = sum / (dataMatrix.Length - 1);
                for (int i = 0; i < dataMatrix.Length; ++i)
                    dataMatrix[i][col] = (dataMatrix[i][col] - mean) / sd;
            }
        }

        public double GetAccuracy(double[][] testData, int[][] testTargets)
        {
            // percentage correct using winner-takes all
            int numCorrect = 0;
            int numWrong = 0;
            double[] xValues = new double[this.numberOfInputs]; // inputs
            int[] tValues = new int[this.numberOfOutputLayerNeurons]; // targets

            for (int i = 0; i < testData.Length; ++i)
            {
                Array.Copy(testData[i], xValues, this.numberOfInputs); // parse test data into x-values and t-values
                Array.Copy(testTargets[i], tValues, this.numberOfOutputLayerNeurons);

                double[] yValues = this.ComputeOutputs(xValues); // computed Y
                int maxIndex = NeuralNetwork.MaxIndex(yValues); // which cell in yValues has largest value?

                if (tValues[maxIndex] == 1)
                    ++numCorrect;
                else
                    ++numWrong;
            }
            return (numCorrect * 1.0) / (numCorrect + numWrong); // ugly 2 - check for divide by zero
        }

        private static int MaxIndex(double[] vector) // helper for Accuracy()
        {
            // index of largest value
            int bigIndex = 0;
            double biggestVal = vector[0];
            for (int i = 0; i < vector.Length; ++i)
            {
                if (vector[i] > biggestVal)
                {
                    biggestVal = vector[i]; bigIndex = i;
                }
            }
            return bigIndex;
        }

        private void Train(int maxEprochs, double learnRate, double momentum)
        {
            this.infoLogger.Log("Start training");

            // train a back-prop style NN classifier using learning rate and momentum
            int epoch = 0;
            double[] xValues = new double[this.numberOfInputs]; // inputs
            double[] targetValues = new double[this.numberOfOutputLayerNeurons]; // target values

            int[] sequence = new int[this.trainingData.Length];
            for (int i = 0; i < sequence.Length; ++i)
                sequence[i] = i;

            while (epoch < maxEprochs)
            {
                if (epoch%(maxEprochs / 10) == 0)
                {
                    this.infoLogger.Log($"Epoch {epoch}");
                }

                double meanSquaredError = this.MeanSquaredError(this.trainingData, this.trainingTargets);
                if (meanSquaredError < 0.001) break; // consider passing value in as parameter

                this.Shuffle(sequence); // visit each training data in random order
                for (int i = 0; i < this.trainingData.Length; ++i)
                {
                    int idx = sequence[i];
                    Array.Copy(this.trainingData[i], xValues, this.numberOfInputs); // get xValues. more flexible would be a 'GetInputsAndTargets()'
                    Array.Copy(this.trainingTargets[i], targetValues, this.numberOfOutputLayerNeurons); // get target values
                    this.ComputeOutputs(xValues); // copy xValues in, compute outputs (and store them internally)
                    this.UpdateWeights(targetValues, learnRate, momentum); // use curr outputs and targets and back-prop to find better weights
                }
                ++epoch;
            }

            this.infoLogger.Log($"Epoch {epoch}");
        }

        private void UpdateWeights(double[] tValues, double learnRate, double momentum)
        {
            // update the weights and biases using back-propagation, with target values, eta (learning rate), alpha (momentum)
            // assumes that SetWeights and ComputeOutputs have been called and so all the internal arrays and matrices have values (other than 0.0)
            if (tValues.Length != this.numberOfOutputLayerNeurons)
                throw new Exception("target values not same Length as output in UpdateWeights");

            // 1. compute output gradients
            for (int i = 0; i < this.outputLayerGradians.Length; ++i)
            {
                double derivative = (1 - this.outputs[i]) * this.outputs[i]; // derivative of softmax = (1 - y) * y (same as log-sigmoid)
                this.outputLayerGradians[i] = derivative * (tValues[i] - this.outputs[i]); // 'mean squared error version' includes (1-y)(y) derivative
                                                                    //oGrads[i] = (tValues[i] - outputs[i]); // cross-entropy version drops (1-y)(y) term! See http://www.cs.mcgill.ca/~dprecup/courses/ML/Lectures/ml-lecture05.pdf page 25.
            }

            // 2. compute hidden gradients
            for (int i = 0; i < this.hiddenLayerGradians.Length; ++i)
            {
                double derivative = (1 - this.hiddenLayerOutputs[i]) * (1 + this.hiddenLayerOutputs[i]); // derivative of tanh = (1 - y) * (1 + y)
                double sum = 0.0;
                for (int j = 0; j < this.numberOfOutputLayerNeurons; ++j) // each hidden delta is the sum of numOutput terms
                {
                    double x = this.outputLayerGradians[j] * this.hiddenLayerToOutputLayerWeights[i][j];
                    sum += x;
                }
                this.hiddenLayerGradians[i] = derivative * sum;
            }

            // 3a. update hidden weights (gradients must be computed right-to-left but weights can be updated in any order)
            for (int i = 0; i < this.inputLayerTohiddenLayerWeights.Length; ++i) // 0..2 (3)
            {
                for (int j = 0; j < this.inputLayerTohiddenLayerWeights[0].Length; ++j) // 0..3 (4)
                {
                    double delta = learnRate * this.hiddenLayerGradians[j] * this.inputs[i]; // compute the new delta
                    this.inputLayerTohiddenLayerWeights[i][j] += delta; // update. note we use '+' instead of '-'. this can be very tricky.
                    this.inputLayerTohiddenLayerWeights[i][j] += momentum * this.inputLayerToHiddenLayer_PreviousWeightsDeltas[i][j]; // add momentum using previous delta. on first pass old value will be 0.0 but that's OK.
                    this.inputLayerToHiddenLayer_PreviousWeightsDeltas[i][j] = delta; // don't forget to save the delta for momentum 
                }
            }

            // 3b. update hidden biases
            for (int i = 0; i < this.hiddenLayerBiases.Length; ++i)
            {
                double delta = learnRate * this.hiddenLayerGradians[i] * 1.0; // the 1.0 is the constant input for any bias; could leave out
                this.hiddenLayerBiases[i] += delta;
                this.hiddenLayerBiases[i] += momentum * this.hiddenLayerPreviousBiasDeltas[i]; // momentum
                this.hiddenLayerPreviousBiasDeltas[i] = delta; // don't forget to save the delta
            }

            // 4. update hidden-output weights
            for (int i = 0; i < this.hiddenLayerToOutputLayerWeights.Length; ++i)
            {
                for (int j = 0; j < this.hiddenLayerToOutputLayerWeights[0].Length; ++j)
                {
                    // see above: hOutputs are inputs to the nn outputs
                    double delta = learnRate * this.outputLayerGradians[j] * this.hiddenLayerOutputs[i];  
                    this.hiddenLayerToOutputLayerWeights[i][j] += delta;
                    this.hiddenLayerToOutputLayerWeights[i][j] += momentum * this.hiddenLayerToOutputLayer_PreviousWeightsDeltas[i][j]; // momentum
                    this.hiddenLayerToOutputLayer_PreviousWeightsDeltas[i][j] = delta; // save
                }
            }

            // 4b. update output biases
            for (int i = 0; i < this.outputLayerBiases.Length; ++i)
            {
                double delta = learnRate * this.outputLayerGradians[i] * 1.0;
                this.outputLayerBiases[i] += delta;
                this.outputLayerBiases[i] += momentum * this.outputLayerPreviousBiasDeltas[i]; // momentum
                this.outputLayerPreviousBiasDeltas[i] = delta; // save
            }
        } // UpdateWeights

        #region ComputeOutputs

        private double[] ComputeOutputs(double[] xValues)
        {
            if (xValues.Length != this.numberOfInputs)
                throw new Exception("Bad xValues array length");

            double[] hiddenSums = new double[this.numberOfHiddenLayerNeurons]; // hidden nodes sums scratch array
            double[] outputSums = new double[this.numberOfOutputLayerNeurons]; // output nodes sums

            for (int i = 0; i < xValues.Length; ++i) // copy x-values to inputs
                this.inputs[i] = xValues[i];

            for (int j = 0; j < this.numberOfHiddenLayerNeurons; ++j)  // compute i-h sum of weights * inputs
                for (int i = 0; i < this.numberOfInputs; ++i)
                    hiddenSums[j] += this.inputs[i] * this.inputLayerTohiddenLayerWeights[i][j]; // note +=

            for (int i = 0; i < this.numberOfHiddenLayerNeurons; ++i)  // add biases to input-to-hidden sums
                hiddenSums[i] += this.hiddenLayerBiases[i];

            for (int i = 0; i < this.numberOfHiddenLayerNeurons; ++i)   // apply activation
                this.hiddenLayerOutputs[i] = NeuralNetwork.HyperTanFunction(hiddenSums[i]); // hard-coded

            for (int j = 0; j < this.numberOfOutputLayerNeurons; ++j)   // compute h-o sum of weights * hOutputs
                for (int i = 0; i < this.numberOfHiddenLayerNeurons; ++i)
                    outputSums[j] += this.hiddenLayerOutputs[i] * this.hiddenLayerToOutputLayerWeights[i][j];

            for (int i = 0; i < this.numberOfOutputLayerNeurons; ++i)  // add biases to input-to-hidden sums
                outputSums[i] += this.outputLayerBiases[i];

            double[] softOut = NeuralNetwork.Softmax(outputSums); // softmax activation does all outputs at once for efficiency
            Array.Copy(softOut, this.outputs, softOut.Length);

            double[] retResult = new double[this.numberOfOutputLayerNeurons]; // could define a GetOutputs method instead
            Array.Copy(this.outputs, retResult, retResult.Length);
            return retResult;
        } // ComputeOutputs

        private static double HyperTanFunction(double x)
        {
            if (x < -20.0) return -1.0; // approximation is correct to 30 decimals
            else if (x > 20.0) return 1.0;
            else return Math.Tanh(x);
        }

        private static double[] Softmax(double[] outputSums) // does all output nodes at once so scale doesn't have to be re-computed each time
        {
            // determine max output sum
            double max = outputSums[0];
            for (int i = 0; i < outputSums.Length; ++i)
                if (outputSums[i] > max) max = outputSums[i];

            // determine scaling factor -- sum of exp(each val - max)
            double scale = 0.0;
            for (int i = 0; i < outputSums.Length; ++i)
                scale += Math.Exp(outputSums[i] - max);

            double[] result = new double[outputSums.Length];
            for (int i = 0; i < outputSums.Length; ++i)
                result[i] = Math.Exp(outputSums[i] - max) / scale;

            return result; // now scaled so that xi sum to 1.0
        }

        #endregion ComputeOutputs

        private void Shuffle(int[] sequence)
        {
            for (int i = 0; i < sequence.Length; ++i)
            {
                int r = this.random.Next(i, sequence.Length);
                int tmp = sequence[r];
                sequence[r] = sequence[i];
                sequence[i] = tmp;
            }
        }

        private double MeanSquaredError(double[][] trainingData, int[][] trainingTargets)
        {
            // average squared error per training tuple
            double sumSquaredError = 0.0;
            double[] xValues = new double[this.numberOfInputs]; // first numInput values in trainData
            double[] tValues = new double[this.numberOfOutputLayerNeurons]; // last numOutput values

            for (int i = 0; i < trainingData.Length; ++i) // walk thru each training case.
            {
                Array.Copy(trainingData[i], xValues, this.numberOfInputs); // get xValues. more flexible would be a 'GetInputsAndTargets()'
                Array.Copy(trainingTargets[i], tValues, this.numberOfOutputLayerNeurons); // get target values
                double[] yValues = this.ComputeOutputs(xValues); // compute output using current weights
                for (int j = 0; j < this.numberOfOutputLayerNeurons; ++j)
                {
                    double err = tValues[j] - yValues[j];
                    sumSquaredError += err * err;
                }
            }

            return sumSquaredError / trainingData.Length;
        }

        private void MakeTrainTestData(Data data)
        {
            int[] sequence = new int[data.Elements.Length]; // create a random sequence of indexes
            for (int i = 0; i < sequence.Length; ++i)
                sequence[i] = i;

            for (int i = 0; i < sequence.Length; ++i)
            {
                int r = this.random.Next(i, sequence.Length);
                int tmp = sequence[r];
                sequence[r] = sequence[i];
                sequence[i] = tmp;
            }

            int trainingRowCount = (int)(data.Elements.Length*0.8);

            this.trainingData = new double[trainingRowCount][];
            this.trainingTargets = new int[trainingRowCount][];

            this.testData = new double[data.Elements.Length - trainingRowCount][];
            this.testTargets = new int[data.Elements.Length - trainingRowCount][];

            int si = 0; // index into sequence[]
            int j = 0; // index into trainData or testData

            for (; si < trainingRowCount; ++si) // first rows to train data
            {
                this.trainingData[j] = new double[data.NumberOfElementsPerSet];
                this.trainingTargets[j] = new int[data.NumberOfDiscreteTargetValues];

                int idx = sequence[si];
                Array.Copy(data.Elements[idx], this.trainingData[j], data.NumberOfElementsPerSet);
                Array.Copy(data.Targets[idx], this.trainingTargets[j], data.NumberOfDiscreteTargetValues);
                ++j;
            }

            j = 0; // reset to start of test data
            for (; si < data.Elements.Length; ++si) // remainder to test data
            {
                this.testData[j] = new double[data.NumberOfElementsPerSet];
                this.testTargets[j] = new int[data.NumberOfDiscreteTargetValues];

                int idx = sequence[si];
                Array.Copy(data.Elements[idx], this.testData[j], data.NumberOfElementsPerSet);
                Array.Copy(data.Targets[idx], this.testTargets[j], data.NumberOfDiscreteTargetValues);
                ++j;
            }
        }

        private static double[][] MakeMatrix(int rows, int columns)
        {
            var result = new double[rows][];
            for (int r = 0; r < result.Length; ++r)
                result[r] = new double[columns];
            return result;
        }

        private void AllocateSpace()
        {
            this.inputs = new double[this.numberOfInputs];

            this.inputLayerTohiddenLayerWeights = NeuralNetwork.MakeMatrix(this.numberOfInputs,
                this.numberOfHiddenLayerNeurons);

            this.hiddenLayerBiases = new double[this.numberOfHiddenLayerNeurons];
            this.hiddenLayerOutputs = new double[this.numberOfHiddenLayerNeurons];

            this.hiddenLayerToOutputLayerWeights = NeuralNetwork.MakeMatrix(this.numberOfHiddenLayerNeurons,
                this.numberOfOutputLayerNeurons);

            this.outputLayerBiases = new double[this.numberOfOutputLayerNeurons];
            this.outputs = new double[this.numberOfOutputLayerNeurons];

            this.hiddenLayerGradians = new double[this.numberOfHiddenLayerNeurons];
            this.outputLayerGradians = new double[this.numberOfOutputLayerNeurons];

            this.inputLayerToHiddenLayer_PreviousWeightsDeltas = NeuralNetwork.MakeMatrix(this.numberOfInputs,
                this.numberOfHiddenLayerNeurons);

            this.hiddenLayerPreviousBiasDeltas = new double[this.numberOfHiddenLayerNeurons];

            this.hiddenLayerToOutputLayer_PreviousWeightsDeltas =
                NeuralNetwork.MakeMatrix(this.numberOfHiddenLayerNeurons, this.numberOfOutputLayerNeurons);

            this.outputLayerPreviousBiasDeltas = new double[this.numberOfOutputLayerNeurons];
        }

        private void InitializeWeights()
        {
            // initialize weights and biases to small random values
            int numWeights = (this.numberOfInputs*this.numberOfHiddenLayerNeurons) +
                             (this.numberOfHiddenLayerNeurons*this.numberOfOutputLayerNeurons) +
                             this.numberOfHiddenLayerNeurons + this.numberOfOutputLayerNeurons;

            var initialWeights = new double[numWeights];

            for (int i = 0; i < initialWeights.Length; ++i)
                initialWeights[i] = (0.001 - 0.0001) * this.random.NextDouble() + 0.0001;

            this.SetWeights(initialWeights);
        }

        private void SetWeights(double[] weights)
        {
            // copy weights and biases in weights[] array to i-h weights, i-h biases, h-o weights, h-o biases

            int k = 0; // points into weights param

            for (int i = 0; i < this.numberOfInputs; ++i)
                for (int j = 0; j < this.numberOfHiddenLayerNeurons; ++j)
                    this.inputLayerTohiddenLayerWeights[i][j] = weights[k++];

            for (int i = 0; i < this.numberOfHiddenLayerNeurons; ++i)
                this.hiddenLayerBiases[i] = weights[k++];

            for (int i = 0; i < this.numberOfHiddenLayerNeurons; ++i)
                for (int j = 0; j < this.numberOfOutputLayerNeurons; ++j)
                    this.hiddenLayerToOutputLayerWeights[i][j] = weights[k++];

            for (int i = 0; i < this.numberOfOutputLayerNeurons; ++i)
                this.outputLayerBiases[i] = weights[k++];
        }
    }

    public class NeuralNetworkException : Exception
    {
        public NeuralNetworkException(string message) : base(message)
        {
        }
    }
}
