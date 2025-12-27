import { Link, Outlet, useLocation } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../lib/api';

export function Layout() {
  const location = useLocation();
  
  const { data: status } = useQuery({
    queryKey: ['status'],
    queryFn: () => apiClient.getStatus(),
    refetchInterval: 60000,
  });

  const navItems = [
    { path: '/', label: 'Home', icon: '🏠' },
    { path: '/dashboard', label: 'Dashboard', icon: '📊' },
    { path: '/upcoming', label: 'Upcoming', icon: '🏁' },
    { path: '/chat', label: 'Chat', icon: '💬', badge: 'Soon' },
  ];

  const isActive = (path: string) => {
    if (path === '/') return location.pathname === '/';
    return location.pathname.startsWith(path);
  };

  return (
    <div className="min-h-screen">
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-f1-dark/90 backdrop-blur-md border-b border-gray-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <Link to="/" className="flex items-center gap-3 group">
              <div className="w-10 h-10 bg-f1-red rounded-lg flex items-center justify-center 
                            transform group-hover:scale-110 transition-transform">
                <span className="text-white font-racing font-bold text-lg">F1</span>
              </div>
              <div className="hidden sm:block">
                <span className="font-racing text-xl text-white">PREDICTOR</span>
                <span className="block text-xs text-gray-500">2026 Season Ready</span>
              </div>
            </Link>

            {/* Nav Links */}
            <div className="flex items-center gap-1 sm:gap-2">
              {navItems.map((item) => (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`
                    relative px-3 sm:px-4 py-2 rounded-lg transition-all duration-200
                    flex items-center gap-2 text-sm font-medium
                    ${isActive(item.path) 
                      ? 'bg-f1-red text-white' 
                      : 'text-gray-400 hover:text-white hover:bg-white/5'}
                  `}
                >
                  <span className="hidden sm:inline">{item.icon}</span>
                  <span>{item.label}</span>
                  {item.badge && (
                    <span className="absolute -top-1 -right-1 bg-yellow-500 text-black text-[10px] 
                                   px-1.5 py-0.5 rounded-full font-bold">
                      {item.badge}
                    </span>
                  )}
                </Link>
              ))}
            </div>

            {/* Status Indicator */}
            <div className="hidden md:flex items-center gap-3">
              <div className={`
                flex items-center gap-2 px-3 py-1.5 rounded-full text-xs
                ${status?.model_exists 
                  ? 'bg-green-500/20 text-green-400 border border-green-500/30' 
                  : 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30'}
              `}>
                <div className={`w-2 h-2 rounded-full ${status?.model_exists ? 'bg-green-400' : 'bg-yellow-400'} animate-pulse`} />
                {status?.model_exists ? 'Model Ready' : 'Setup Required'}
              </div>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="pt-16 min-h-screen">
        <Outlet />
      </main>

      {/* Footer */}
      <footer className="border-t border-gray-800 py-8 mt-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
            <div className="text-gray-500 text-sm">
              F1 Race Prediction System — Built with LightGBM & FastF1
            </div>
            <div className="flex items-center gap-4 text-sm text-gray-500">
              <span>Data: 2018-2025</span>
              <span>•</span>
              <span>{status?.available_years?.length || 0} seasons</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
