using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BackPropagationNetwork
{
    public class Data
    {
        public readonly double[][] Elements;
        public readonly int[][] Targets;

        public readonly int NumberOfElementsPerSet;
        public readonly int NumberOfDiscreteTargetValues;

        private ulong dataArrayPointer = 0;
        private int currentTargetIndex = 0;

        private readonly ConcurrentDictionary<int, int[]> targetEncodingDictionary = new ConcurrentDictionary<int, int[]>();

        private readonly ConcurrentDictionary<int[], int> targetDecodingDictionary =
            new ConcurrentDictionary<int[], int>(new IntArrayEqualityComparer());

        public Data(ulong numberOfElementSets, int numberOfElementsPerSet, int numberOfDiscreteTargetValues)
        {
            this.Elements = new double[numberOfElementSets][];
            this.Targets = new int[numberOfElementSets][];

            this.NumberOfElementsPerSet = numberOfElementsPerSet;
            this.NumberOfDiscreteTargetValues = numberOfDiscreteTargetValues;
        }

        public int DecodeTarget(int[] input)
        {
            int target;
            if (this.targetDecodingDictionary.TryGetValue(input, out target))
            {
                return target;
            }

            var sb = new StringBuilder();

            foreach (int i in input)
            {
                sb.Append(i + ", ");
            }

            if (input.Length > 0)
            {
                sb.Remove(sb.Length - 2, 2);
            }

            throw new DataException($"Encoded value not found: [{sb}]");
        }

        private class IntArrayEqualityComparer : IEqualityComparer<int[]>
        {
            public bool Equals(int[] x, int[] y)
            {
                if (x.Length != y.Length)
                {
                    return false;
                }

                return !x.Where((t, i) => t != y[i]).Any();
            }

            public int GetHashCode(int[] obj)
            {
                return obj.Aggregate(0, (current, t) => current ^ t.GetHashCode());
            }
        }

        public void Add(double[] elements, int target)
        {
            if (elements.Length != this.NumberOfElementsPerSet)
            {
                throw new DataException(
                    "Input element count does not match initial value. " +
                    $"Initial value {this.Elements.Length} " +
                    $"Input element count: {elements.Length} " +
                    $"Element # {this.dataArrayPointer}");
            }

            this.Elements[this.dataArrayPointer] = elements;

            this.Targets[this.dataArrayPointer] = this.targetEncodingDictionary.GetOrAdd(target, t =>
            {
                var targetResult = new int[this.NumberOfDiscreteTargetValues];
                targetResult[this.currentTargetIndex++] = 1;

                this.targetDecodingDictionary.TryAdd(targetResult, t);

                return targetResult;
            });

            this.dataArrayPointer++;
        }
    }

    public class DataException : Exception
    {
        public DataException(string message) : base(message)
        {
        }
    }
}
