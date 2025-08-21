import { useEffect, useState, useCallback } from 'react';
import { useConfig } from '../../ConfigContext';
import { getApiUrl } from '../../../config';

interface ToolSelectionStrategy {
  key: string;
  label: string;
  description: string;
  enabled: boolean;
}

interface Features {
  "vectordb-sqlite": boolean;
}

export const getAvailableStrategies = (features: Features): ToolSelectionStrategy[] => [
  {
    key: 'llm',
    label: 'LLM-based',
    description: 'Use LLM-based intelligence to select the most relevant tools based on the user query context.',
    enabled: true,
  },
  {
    key: 'vector',
    label: 'Vector-based',
    description: 'Use vector similarity search to quickly find the most relevant tools.',
    enabled: features["vectordb-sqlite"],
  },
];

export const ToolSelectionStrategySection = () => {
  const [currentStrategy, setCurrentStrategy] = useState<string>('llm');
  const [availableStrategies, setAvailableStrategies] = useState<ToolSelectionStrategy[]>([]);
  const [_error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const { read, upsert } = useConfig();

  const handleStrategyChange = async (strategy: string) => {
    if (isLoading) return; // Prevent multiple simultaneous requests

    setError(null); // Clear any previous errors
    setIsLoading(true);

    try {
      // First update the configuration
      try {
        await upsert('GOOSE_TOOL_SELECTION_STRATEGY', strategy, false);
      } catch (error) {
        console.error('Error updating configuration:', error);
        setError(`Failed to update configuration: ${error}`);
        setIsLoading(false);
        return;
      }

      // Then update the backend
      try {
        const response = await fetch(getApiUrl('/agent/update_router_tool_selector'), {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-Secret-Key': await window.electron.getSecretKey(),
          },
        });

        if (!response.ok) {
          const errorData = await response
            .json()
            .catch(() => ({ error: 'Unknown error from backend' }));
          throw new Error(errorData.error || 'Unknown error from backend');
        }

        // Parse the success response
        const data = await response
          .json()
          .catch(() => ({ message: 'Tool selection strategy updated successfully' }));
        if (data.error) {
          throw new Error(data.error);
        }
      } catch (error) {
        console.error('Error updating backend:', error);
        setError(`Failed to update backend: ${error}`);
        setIsLoading(false);
        return;
      }

      // If both succeeded, update the UI
      setCurrentStrategy(strategy);
    } catch (error) {
      console.error('Error updating tool selection strategy:', error);
      setError(`Failed to update tool selection strategy: ${error}`);
    } finally {
      setIsLoading(false);
    }
  };

  const fetchCurrentStrategy = useCallback(async () => {
    try {
      const strategy = (await read('GOOSE_TOOL_SELECTION_STRATEGY', false)) as string;
      if (strategy) {
        setCurrentStrategy(strategy);
      } else {
        // Default to LLM if no strategy is set
        setCurrentStrategy('llm');
      }
    } catch (error) {
      console.error('Error fetching current tool selection strategy:', error);
      setError(`Failed to fetch current tool selection strategy: ${error}`);
    }
  }, [read]);

  const fetchAvailableFeatures = useCallback(async () => {
    try {
      const response = await fetch(getApiUrl('/features'));
      if (!response.ok) {
        throw new Error('Failed to fetch features');
      }
      const data = await response.json();
      const strategies = getAvailableStrategies(data.features);
      setAvailableStrategies(strategies);
    } catch (error) {
      console.error('Error fetching available features:', error);
      // Fallback to LLM only if features endpoint fails
      setAvailableStrategies(getAvailableStrategies({ "vectordb-sqlite": false }));
    }
  }, []);

  useEffect(() => {
    fetchCurrentStrategy();
    fetchAvailableFeatures();
  }, [fetchCurrentStrategy, fetchAvailableFeatures]);

  // Only render the component if there are multiple enabled strategies
  const enabledStrategies = availableStrategies.filter(strategy => strategy.enabled);
  
  if (enabledStrategies.length <= 1) {
    return null; // Don't render anything if there's only one or no strategies available
  }

  return (
    <div className="space-y-1">
      {availableStrategies.map((strategy) => (
        <div 
          className={`group ${strategy.enabled ? 'hover:cursor-pointer' : 'cursor-not-allowed opacity-50'}`} 
          key={strategy.key}
        >
          <div
            className={`flex items-center justify-between text-text-default py-2 px-2 ${
              currentStrategy === strategy.key 
                ? 'bg-background-muted' 
                : strategy.enabled 
                  ? 'bg-background-default hover:bg-background-muted' 
                  : 'bg-background-default'
            } rounded-lg transition-all`}
            onClick={() => strategy.enabled && handleStrategyChange(strategy.key)}
          >
            <div className="flex">
              <div>
                <h3 className="text-text-default text-xs">
                  {strategy.label}
                  {!strategy.enabled && " (Unavailable)"}
                </h3>
                <p className="text-xs text-text-muted mt-[2px]">
                  {strategy.enabled 
                    ? strategy.description 
                    : "This strategy requires additional features to be enabled."
                  }
                </p>
              </div>
            </div>

            <div className="relative flex items-center gap-2">
              <input
                type="radio"
                name="tool-selection-strategy"
                value={strategy.key}
                checked={currentStrategy === strategy.key}
                onChange={() => strategy.enabled && handleStrategyChange(strategy.key)}
                disabled={isLoading || !strategy.enabled}
                className="peer sr-only"
              />
              <div
                className="h-4 w-4 rounded-full border border-border-default
                      peer-checked:border-[6px] peer-checked:border-black dark:peer-checked:border-white
                      peer-checked:bg-white dark:peer-checked:bg-black
                      transition-all duration-200 ease-in-out group-hover:border-border-default"
              ></div>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
};
