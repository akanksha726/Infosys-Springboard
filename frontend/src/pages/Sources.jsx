import { FileText, ExternalLink, Zap, Search } from 'lucide-react';

export default function Sources({ data }) {
  if (!data) return <div className="p-8 animated-enter text-center">Loading Research Sources...</div>;

  const sources = data?.consumer_insights?.sources || [];

  return (
    <div className="p-8 animated-enter">
      <div className="flex justify-between items-center mb-8">
        <div>
          <h1 className="text-3xl font-bold text-white mb-2">Research Sources & Reference</h1>
          <p className="text-secondary">Explore the original source material and news articles driving the AI market analysis.</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {sources.map((source, idx) => (
            <div key={idx} className="glass-panel p-6 flex flex-col gap-4 border border-white/5 hover:border-blue-500/30 transition-all group">
              <div className="flex justify-between items-start">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-blue-500/10 rounded-lg">
                    <Zap size={20} className="text-blue-400" />
                  </div>
                  <div>
                    <h3 className="font-bold text-white">Source {source.source_id || idx + 1}</h3>
                    <span className="text-[10px] text-secondary uppercase tracking-widest opacity-60">
                      RAG Verified Document
                    </span>
                  </div>
                </div>
                <a 
                  href={source.url} 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="p-2 bg-white/5 hover:bg-white/10 rounded-full transition-colors text-secondary hover:text-white"
                  title="Open Article"
                >
                  <ExternalLink size={18} />
                </a>
              </div>

              <div className="flex-1">
                <div className="text-sm text-secondary leading-relaxed italic bg-black/20 p-4 rounded-lg border border-white/5">
                  "{source.content}"
                </div>
              </div>

              <div className="flex justify-between items-center pt-4 border-t border-white/5">
                <span className="text-[10px] text-secondary font-mono truncate max-w-[200px] opacity-40">
                  {source.url}
                </span>
                <a 
                  href={source.url} 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="text-xs font-bold text-blue-400 hover:text-blue-300 transition-colors flex items-center gap-2"
                >
                  Read Full Article <ExternalLink size={12} />
                </a>
              </div>
            </div>
        ))}

        {sources.length === 0 && (
          <div className="col-span-full text-center p-12 glass-panel opacity-50">
            <div className="mb-4 flex justify-center">
              <FileText size={48} className="text-secondary" />
            </div>
            <p className="text-lg">No research sources currently available.</p>
            <p className="text-sm">Run the market analysis engine to generate new insights.</p>
          </div>
        )}
      </div>

      <div className="mt-12 flex justify-center pb-12">
        <a 
          href="https://news.google.com/search?q=ecommerce+market+india+trends" 
          target="_blank" 
          rel="noopener noreferrer"
          className="glass-btn flex items-center gap-3 px-8 py-4 text-blue-400 border border-blue-500/30 hover:bg-blue-500/10 transition-all font-bold group"
        >
          <Search size={18} className="group-hover:scale-110 transition-transform" />
          <span>Discover More Market News</span>
          <ExternalLink size={16} />
        </a>
      </div>
    </div>
  );
}
