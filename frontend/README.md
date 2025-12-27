# F1 Race Predictor - Frontend

Modern React + TypeScript frontend for the F1 Race Prediction System.

## Tech Stack

- **React 18** + **TypeScript**
- **Vite** - Fast build tool
- **Tailwind CSS** - Utility-first styling
- **TanStack React Query** - Data fetching and caching
- **React Router** - Client-side routing
- **Recharts** - Data visualization

## Setup

### Install Dependencies

```bash
npm install
```

### Environment Configuration

Create a `.env` file in the frontend directory:

```env
VITE_API_URL=http://localhost:8000
```

### Development

```bash
npm run dev
```

The app will be available at `http://localhost:5173`

### Build for Production

```bash
npm run build
```

### Preview Production Build

```bash
npm run preview
```

## Pages

### Dashboard (`/`)

- System status overview
- Model performance metrics
- Quick start guide
- Data availability

### Predict (`/predict`)

- Select year and race
- View predicted finishing order
- Visualize grid vs predicted positions
- Podium predictions highlighted
- Model confidence scores

### Update (`/update`)

- Post-race data fetching
- Optional model retraining
- Real-time update status
- Updated metrics display

## Features

### Pre-Race Predictions

- Uses only data available before race start
- Qualifying results
- Driver/team historical performance
- Track characteristics
- Weather conditions

### Post-Race Updates

- Fetches race results from FastF1 API
- Rebuilds features with new data
- Retrains model for improved future predictions
- Rolling improvement throughout the season

### UI/UX

- Dark mode motorsport theme
- Responsive design (mobile-friendly)
- Loading states and error handling
- Skeleton loaders for better perceived performance
- F1-inspired color scheme (red accent on dark)

## API Integration

The frontend communicates with the FastAPI backend at `http://localhost:8000` (configurable via `.env`).

### Endpoints Used

- `GET /status` - System status and model info
- `GET /races/{year}` - Available races for a year
- `GET /predict?year={year}&round={round}` - Race predictions
- `POST /update?year={year}&round={round}&retrain={bool}` - Post-race update

## Development Notes

### State Management

- React Query handles all server state
- Local component state for UI interactions
- No global state library needed

### Styling

- Tailwind utility classes
- Custom F1 color palette in `tailwind.config.js`
- Component-level styles in `index.css`

### Type Safety

- Full TypeScript coverage
- API response types in `src/types/index.ts`
- Type-safe API client in `src/lib/api.ts`

## Troubleshooting

### API Connection Issues

Ensure the FastAPI backend is running:

```bash
cd ..
python run_api.py
```

### CORS Errors

The backend should allow `http://localhost:5173` origin. Check `backend/api.py` CORS configuration.

### Build Errors

Clear node_modules and reinstall:

```bash
rm -rf node_modules package-lock.json
npm install
```

## Future Enhancements

- [ ] Live race updates during race weekends
- [ ] Historical comparison charts
- [ ] Driver/constructor deep-dive pages
- [ ] Chat assistant with RAG pipeline
- [ ] Export predictions to CSV
- [ ] Mobile app version

