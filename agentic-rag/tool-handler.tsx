// agentic-rag/tool-handler.tsx
// Local tool handlers for agentic flows

export type NutritionPlannerArgs = {
  run_block?: string;
  what_to_cover?: string;
  context?: string;
};

export type WeatherCheckerArgs = {
  location?: string;
  datetime?: string;
  timezone?: string;
  duration_minutes?: number;
};

export type ToolResult = {
  name: string;
  summary: string;
  hydrationPlan?: {
    preRun?: string;
    duringRun?: string;
    electrolytes?: string;
  };
  weatherData?: {
    temperature?: number;
    humidity?: number;
    conditions?: string;
    windSpeed?: number;
    uvIndex?: number;
  };
  recommendations?: string[];
  notes?: string[];
};

export async function nutrition_planner(args: NutritionPlannerArgs): Promise<ToolResult> {
  const run = (args.run_block || '').trim();
  const cover = (args.what_to_cover || '').trim();
  const ctx = (args.context || '').trim();

  // Extremely simple heuristic output for demo purposes
  const preRun = 'Drink 200–300 ml water 20–30 min before start.';
  const during = 'Sip 100–200 ml every 15–20 min; adjust for sweat rate and conditions.';
  const electrolytes = 'If sweat rate is moderate, add 200–300 mg sodium per hour via drink/tablet.';

  const notes: string[] = [];
  if (/ankle/i.test(ctx)) notes.push('Avoid uneven terrain; prefer flat, stable surfaces.');
  if (/15\s*°?C/i.test(ctx)) notes.push('Cool 15°C: total fluids ~400–700 ml/hour is often sufficient.');

  const summary = `Hydration & electrolytes plan for: ${run || 'the recovery run'}. Covers: ${cover || 'hydration focus'}.`;

  return {
    name: 'nutrition_planner',
    summary,
    hydrationPlan: {
      preRun,
      duringRun: during,
      electrolytes,
    },
    notes,
  };
}

export async function weather_checker(args: WeatherCheckerArgs): Promise<ToolResult> {
  const location = (args.location || 'Unknown location').trim();
  const datetime = (args.datetime || new Date().toISOString()).trim();
  const duration = args.duration_minutes || 60;

  // Mock weather data for demo purposes
  // In production, this would call a weather API like OpenWeatherMap
  const temperature = 15; // °C
  const humidity = 65; // %
  const conditions = 'Clear';
  const windSpeed = 8; // km/h
  const uvIndex = 3;

  const recommendations: string[] = [];
  
  // Temperature-based recommendations
  if (temperature < 10) {
    recommendations.push('Cool conditions: warm up thoroughly; layer clothing for initial chill.');
  } else if (temperature > 25) {
    recommendations.push('Warm conditions: increase fluid intake by 20-30%; consider earlier start time.');
  }

  // Humidity adjustments
  if (humidity > 70) {
    recommendations.push('High humidity: expect reduced sweat evaporation; increase electrolyte intake.');
  }

  // Wind considerations
  if (windSpeed > 15) {
    recommendations.push('Moderate wind: adjust pace on windy segments; protect against wind chill.');
  }

  // UV protection
  if (uvIndex > 6) {
    recommendations.push('High UV: apply sunscreen; wear cap or visor for sun protection.');
  }

  const summary = `Weather for ${location} at ${datetime}: ${temperature}°C, ${conditions}, ${humidity}% humidity. Duration: ${duration} min.`;

  return {
    name: 'weather_checker',
    summary,
    weatherData: {
      temperature,
      humidity,
      conditions,
      windSpeed,
      uvIndex,
    },
    recommendations,
  };
}
