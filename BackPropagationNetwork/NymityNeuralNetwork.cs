using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BackPropagationNetwork.DebugLogger;

namespace BackPropagationNetwork
{
    public class NymityNeuralNetwork : NeuralNetwork
    {
        public NymityNeuralNetwork() : base(new ConsoleDebugLogger())
        {
        }
    }
}
