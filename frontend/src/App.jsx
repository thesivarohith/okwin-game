import { useState, useEffect } from 'react'

function App() {
  const [history, setHistory] = useState(Array(20).fill(0))
  const [period, setPeriod] = useState("----")
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [stats, setStats] = useState(null)
  const [autoSync, setAutoSync] = useState(false)
  const [clearedAt, setClearedAt] = useState(null)
  const [lastSync, setLastSync] = useState(null)

  const fetchStats = async () => {
    try {
      const res = await fetch('http://localhost:8000/stats')
      if (res.ok) {
        const data = await res.json()
        setStats(data)
      }
    } catch (e) {
      console.error('Failed to fetch stats', e)
    }
  }

  useEffect(() => {
    fetchStats()
  }, [])

  useEffect(() => {
    let interval;
    if (autoSync) {
      const fetchAuto = async () => {
        try {
          const res = await fetch('http://localhost:8000/auto-predict')
          if (res.ok) {
            const data = await res.json()
            setHistory(data.history)
            setPeriod(String(data.target_period))
            setPrediction(data)
            setLastSync(new Date().toLocaleTimeString())
            fetchStats()
          }
        } catch(e) {
          console.error("Auto sync failed", e)
        }
      }
      
      fetchAuto()
      interval = setInterval(fetchAuto, 5000)
    }
    return () => clearInterval(interval)
  }, [autoSync])

  const handleManualPredict = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ history, period })
      })
      if (!res.ok) throw new Error('API Error')
      const data = await res.json()
      setPrediction(data)
      fetchStats()
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const handleAddResult = async (val) => {
    setLoading(true)
    setError(null)
    const newHist = [...history.slice(1), val]
    setHistory(newHist)
    const nextPeriod = (BigInt(period) + 1n).toString()
    
    try {
      const res = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ history: newHist, period: nextPeriod })
      })
      if (!res.ok) throw new Error('API Error')
      const data = await res.json()
      setPrediction(data)
      setPeriod(nextPeriod)
      fetchStats()
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  // Calculate session stats based on visible history
  const getSessionStats = () => {
    if (!stats || !stats.recent_predictions) return { total: 0, wins: 0, rate: 0, pending: 0, gradedTotal: 0 }
    const filtered = stats.recent_predictions.filter(p => !clearedAt || p.timestamp > clearedAt)
    const graded = filtered.filter(p => p.correct !== undefined)
    const pending = filtered.filter(p => p.correct === undefined).length
    const wins = graded.filter(p => p.correct === true).length
    return {
      total: filtered.length,
      gradedTotal: graded.length,
      wins: wins,
      rate: graded.length > 0 ? Math.round((wins / graded.length) * 100) : 0,
      pending: pending
    }
  }

  const sessionStats = getSessionStats()

  return (
    <div className="min-h-screen p-6 md:p-12 max-w-5xl mx-auto flex flex-col gap-8">
      {/* Header */}
      <header className="border-b border-gray-800 pb-6 flex justify-between items-end">
        <div>
          <h1 className="text-3xl font-light tracking-tight text-white mb-2">OkWin Predictor</h1>
          <p className="text-gray-400 text-sm">AI-Powered Ensemble Prediction Engine (Big/Small)</p>
        </div>
        <div className="text-right">
          <button 
            onClick={() => setAutoSync(!autoSync)}
            className={`px-4 py-2 rounded font-medium text-sm transition-colors border ${autoSync ? 'bg-green-900/40 text-green-400 border-green-800' : 'bg-[#111] text-gray-400 border-gray-800 hover:text-white'}`}
          >
            {autoSync ? '🟢 Live Auto-Sync ON' : '⚫ Auto-Sync OFF'}
          </button>
          {lastSync && <p className="text-[10px] text-gray-600 mt-1 uppercase font-mono">Last Sync: {lastSync}</p>}
        </div>
      </header>

      {/* Main Grid */}
      <main className="grid grid-cols-1 md:grid-cols-3 gap-8 items-start">
        <section className="md:col-span-2 flex flex-col gap-6">
          <div className="bg-[#111] border border-gray-800 rounded-xl p-6">
            <h2 className="text-lg font-medium text-white mb-4">Input Sequence</h2>
            <p className="text-xs text-gray-500 mb-2 tracking-wide font-mono uppercase">Current 20 Rounds (Shifts Left Automatically)</p>
            <div className="flex flex-wrap gap-1.5 mb-8">
              {history.map((val, idx) => (
                <div key={idx} className="w-8 h-8 flex items-center justify-center bg-black border border-gray-800 rounded text-xs text-gray-400">
                  {val}
                </div>
              ))}
            </div>

            <p className="text-xs text-blue-400 mb-3 tracking-wide font-mono uppercase">Add the Outcome & Predict</p>
            <div className="grid grid-cols-5 gap-3 mb-6">
              {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map((num) => (
                <button
                  key={num}
                  onClick={() => handleAddResult(num)}
                  disabled={loading}
                  className={`py-3 rounded-md font-mono text-center transition-colors border
                    ${num >= 5 
                      ? 'bg-[#1a0a0a] border-red-900/50 text-red-500 hover:bg-red-900/20' 
                      : 'bg-[#0a1a0f] border-green-900/50 text-green-500 hover:bg-green-900/20'
                    } disabled:opacity-50`}
                >
                  {num}
                </button>
              ))}
            </div>

            <div className="mt-6 pt-6 border-t border-gray-800">
              <label className="block text-xs text-gray-500 mb-2 font-mono uppercase">Target Period ID (Auto-Increments)</label>
              <div className="flex gap-3">
                <input 
                  type="text" 
                  value={period}
                  onChange={(e) => setPeriod(e.target.value.replace(/[^0-9]/g, ''))}
                  className="flex-1 bg-black border border-gray-800 rounded-md py-3 px-4 text-white focus:outline-none focus:border-gray-500 font-mono"
                />
                <button 
                  onClick={handleManualPredict}
                  disabled={loading}
                  className="bg-white text-black font-medium py-3 px-6 rounded-md hover:bg-gray-200 transition-colors disabled:opacity-50"
                >
                  Predict
                </button>
              </div>
            </div>
            {error && <p className="text-red-400 text-sm mt-4 font-mono">{error}</p>}
          </div>

          {prediction ? (
            <div className="bg-[#111] p-6 rounded-lg border border-gray-800 shadow-xl">
              <span className="text-xs text-blue-400 mb-2 uppercase block tracking-wider font-bold">Analysis Result</span>
              <div className="flex justify-between items-end mb-6">
                <div>
                  <h2 className="text-6xl font-light text-white">{prediction.prediction}</h2>
                  <p className="text-gray-500 text-sm mt-2 font-mono">Target Period: {String(prediction.target_period || '')}</p>
                </div>
                <div className="text-right">
                  <span className="text-4xl font-light text-white">{prediction.confidence}%</span>
                  <p className="text-gray-500 text-sm mt-2">Ensemble Confidence</p>
                </div>
              </div>

              <div className="space-y-4">
                <div>
                  <div className="flex justify-between text-[10px] uppercase tracking-wider mb-1">
                    <span className="text-gray-400">Big Probability</span>
                    <span className="text-white font-mono">{prediction.probabilities.Big}%</span>
                  </div>
                  <div className="h-1 bg-gray-900 rounded-full overflow-hidden">
                    <div className="h-full bg-white transition-all duration-1000" style={{ width: `${prediction.probabilities.Big}%` }}></div>
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-[10px] uppercase tracking-wider mb-1">
                    <span className="text-gray-400">Small Probability</span>
                    <span className="text-white font-mono">{prediction.probabilities.Small}%</span>
                  </div>
                  <div className="h-1 bg-gray-900 rounded-full overflow-hidden">
                    <div className="h-full bg-gray-600 transition-all duration-1000" style={{ width: `${prediction.probabilities.Small}%` }}></div>
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-3 gap-4 mt-8 pt-6 border-t border-gray-900">
                <div className="text-center">
                  <span className="text-[10px] text-gray-500 block mb-1 uppercase tracking-tight">XGBoost (60%)</span>
                  <span className="text-xs font-mono text-gray-300">{prediction.model_contributions.xgboost.Big}% Big</span>
                </div>
                <div className="text-center">
                  <span className="text-[10px] text-gray-500 block mb-1 uppercase tracking-tight">Bi-LSTM (20%)</span>
                  <span className="text-xs font-mono text-gray-300">{prediction.model_contributions.lstm.Big}% Big</span>
                </div>
                <div className="text-center">
                  <span className="text-[10px] text-gray-500 block mb-1 uppercase tracking-tight">Markov (20%)</span>
                  <span className="text-xs font-mono text-gray-300">{prediction.model_contributions.markov.Big}% Big</span>
                </div>
              </div>
            </div>
          ) : (
            <div className="bg-[#111] p-12 rounded-lg border border-gray-800 border-dashed flex flex-col items-center justify-center text-center text-gray-600">
              <p>Select a number or wait for Auto-Sync to see result</p>
            </div>
          )}
        </section>

        <aside className="md:col-span-1 flex flex-col gap-6">
          <div className="bg-[#111] border border-gray-800 rounded-xl p-6">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-lg font-medium text-white">Target Stats</h2>
              <button 
                onClick={() => setClearedAt(Date.now() / 1000)}
                className="px-2 py-1 bg-red-900/20 border border-red-900/40 text-[10px] text-red-400 hover:bg-red-900/40 uppercase tracking-widest font-bold transition-all rounded"
                title="Clears session stats and history"
              >
                Reset Stats
              </button>
            </div>
            
            {stats ? (
              <div className="flex flex-col gap-8">
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-[#0a0a0a] p-4 rounded-lg border border-gray-900 text-center">
                    <div className="text-2xl font-light text-white">{sessionStats.total}</div>
                    <div className="text-[10px] text-gray-600 uppercase mt-1 tracking-tighter text-blue-400">Total Predictions</div>
                  </div>
                  <div className="bg-[#0a0a0a] p-4 rounded-lg border border-gray-900 text-center relative overflow-hidden">
                    <div className="text-2xl font-light text-blue-400">{sessionStats.rate}%</div>
                    <div className="text-[10px] text-gray-600 uppercase mt-1 tracking-tighter">Win Rate</div>
                    {sessionStats.pending > 0 && (
                      <div className="absolute top-0 right-0 p-1">
                        <span className="flex h-2 w-2 rounded-full bg-yellow-500 animate-pulse"></span>
                      </div>
                    )}
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-[#0a0a0a] p-3 rounded-lg border border-gray-900 text-center">
                    <div className="text-xl font-light text-green-500">{sessionStats.wins}</div>
                    <div className="text-[10px] text-gray-600 uppercase tracking-tighter">Wins</div>
                  </div>
                  <div className="bg-[#0a0a0a] p-3 rounded-lg border border-gray-900 text-center">
                    <div className="text-xl font-light text-red-500">{sessionStats.total - sessionStats.wins}</div>
                    <div className="text-[10px] text-gray-600 uppercase tracking-tighter">Losses</div>
                  </div>
                </div>

                <div className="space-y-4">
                  <div className="flex justify-between items-center mb-4 border-b border-gray-900 pb-2">
                    <h3 className="text-[10px] text-gray-500 uppercase tracking-widest font-bold">Session History</h3>
                    <span className="text-[10px] text-gray-600 font-mono">W: {sessionStats.wins} / L: {sessionStats.total - sessionStats.wins}</span>
                  </div>
                  
                  <div className="space-y-2 max-h-[400px] overflow-y-auto pr-1 custom-scrollbar">
                    {stats.recent_predictions
                      .filter(p => !clearedAt || p.timestamp > clearedAt)
                      .map((p, i) => (
                        <div key={i} className="flex items-center justify-between p-3 bg-[#0a0a0a] rounded border border-gray-900">
                          <div className="flex items-center gap-3">
                            <span className={`w-1.5 h-1.5 rounded-full ${p.prediction === 'Big' ? 'bg-white' : 'bg-gray-600'}`}></span>
                            <span className="text-xs font-mono text-gray-400">#{p.target_period?.toString()?.slice(-4) || '----'}: {p.prediction}</span>
                          </div>
                          <div className="flex items-center gap-2">
                             {p.correct !== undefined ? (
                               <span className={`text-[10px] font-bold uppercase ${p.correct ? 'text-green-500' : 'text-red-500'}`}>
                                 {p.correct ? 'WIN' : 'LOSS'}
                               </span>
                             ) : (
                               <span className="text-[10px] text-gray-600 uppercase animate-pulse">Wait...</span>
                             )}
                             <span className="text-[10px] text-gray-700 font-mono">{(p.confidence / 10).toFixed(1)}/10</span>
                          </div>
                        </div>
                      ))}
                    {stats.recent_predictions.filter(p => !clearedAt || p.timestamp > clearedAt).length === 0 && (
                      <div className="text-center py-12 text-gray-700 text-[10px] uppercase font-mono border border-dashed border-gray-900 rounded">
                        --- Session Empty ---
                        <br/>
                        <span className="mt-2 block opacity-50 italic">Waiting for new results...</span>
                      </div>
                    )}
                  </div>
                </div>

                <div className="pt-4 border-t border-gray-900">
                    <div className="flex justify-between text-[10px] text-gray-600 uppercase">
                        <span>Overall Database Accuracy</span>
                        <span>{stats.success_rate}%</span>
                    </div>
                </div>
              </div>
            ) : (
              <p className="text-sm text-gray-500 italic">Connecting to engine...</p>
            )}
          </div>
        </aside>
      </main>
    </div>
  )
}

export default App
