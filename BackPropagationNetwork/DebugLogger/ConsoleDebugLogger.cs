using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BackPropagationNetwork.DebugLogger
{
    public class ConsoleDebugLogger : IDebugLogger
    {
        public void Log(string message)
        {
            Console.WriteLine(message);
        }
    }
}
