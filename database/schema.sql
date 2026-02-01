-- Schema para Supabase - Melting Point Prediction
-- Ejecutar este script en el SQL Editor de Supabase

-- =======================
-- TABLA: compounds
-- Almacena todos los compuestos del dataset original
-- =======================
CREATE TABLE IF NOT EXISTS compounds (
    id SERIAL PRIMARY KEY,
    compound_id INTEGER UNIQUE NOT NULL, -- ID original del dataset
    smiles TEXT NOT NULL,
    tm_real DECIMAL(10, 4), -- Temperatura de fusión real (K)
    dataset_type VARCHAR(10) NOT NULL CHECK (dataset_type IN ('train', 'test')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Índices para búsquedas rápidas
CREATE INDEX idx_compounds_smiles ON compounds(smiles);
CREATE INDEX idx_compounds_dataset_type ON compounds(dataset_type);
CREATE INDEX idx_compounds_compound_id ON compounds(compound_id);

-- =======================
-- TABLA: predictions
-- Almacena las predicciones del modelo ChemProp
-- =======================
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    compound_id INTEGER REFERENCES compounds(id) ON DELETE CASCADE,
    smiles TEXT NOT NULL,
    tm_pred DECIMAL(10, 4) NOT NULL, -- Predicción en Kelvin
    model_version VARCHAR(50) DEFAULT 'chemprop_v1',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_predictions_compound_id ON predictions(compound_id);
CREATE INDEX idx_predictions_smiles ON predictions(smiles);

-- =======================
-- TABLA: users (opcional para autenticación futura)
-- =======================
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT UNIQUE,
    username TEXT UNIQUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login TIMESTAMP WITH TIME ZONE
);

-- =======================
-- TABLA: user_predictions
-- Almacena predicciones personalizadas de usuarios
-- =======================
CREATE TABLE IF NOT EXISTS user_predictions (
    id SERIAL PRIMARY KEY,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    smiles TEXT NOT NULL,
    tm_pred DECIMAL(10, 4) NOT NULL,
    session_id TEXT, -- Para usuarios no autenticados
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB -- Para almacenar info adicional flexible
);

CREATE INDEX idx_user_predictions_user_id ON user_predictions(user_id);
CREATE INDEX idx_user_predictions_session_id ON user_predictions(session_id);
CREATE INDEX idx_user_predictions_created_at ON user_predictions(created_at DESC);

-- =======================
-- TABLA: model_metadata
-- Información sobre el modelo ML
-- =======================
CREATE TABLE IF NOT EXISTS model_metadata (
    id SERIAL PRIMARY KEY,
    model_version VARCHAR(50) UNIQUE NOT NULL,
    model_type VARCHAR(50) NOT NULL, -- 'chemprop', 'lgbm', etc.
    mae DECIMAL(10, 4),
    rmse DECIMAL(10, 4),
    r2_score DECIMAL(10, 4),
    training_date TIMESTAMP WITH TIME ZONE,
    num_folds INTEGER,
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Insertar metadata del modelo actual
INSERT INTO model_metadata (
    model_version, 
    model_type, 
    mae, 
    num_folds, 
    description,
    is_active
) VALUES (
    'chemprop_v1',
    'chemprop_d_mpnn',
    28.85,
    5,
    'ChemProp D-MPNN model with 5-fold cross-validation. MAE: 28.85 K (±3.16 K std dev)',
    TRUE
) ON CONFLICT (model_version) DO NOTHING;

-- =======================
-- TABLA: statistics_cache
-- Caché de estadísticas calculadas para mejorar performance
-- =======================
CREATE TABLE IF NOT EXISTS statistics_cache (
    id SERIAL PRIMARY KEY,
    stat_type VARCHAR(50) UNIQUE NOT NULL, -- 'dataset_distribution', 'model_metrics', etc.
    data JSONB NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =======================
-- FUNCIONES Y TRIGGERS
-- =======================

-- Función para actualizar updated_at automáticamente
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger para compounds
CREATE TRIGGER update_compounds_updated_at
    BEFORE UPDATE ON compounds
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- =======================
-- ROW LEVEL SECURITY (RLS)
-- Configurar después de implementar autenticación
-- =======================

-- Habilitar RLS en user_predictions (ejemplo)
-- ALTER TABLE user_predictions ENABLE ROW LEVEL SECURITY;

-- Política: usuarios solo ven sus propias predicciones
-- CREATE POLICY "Users can view own predictions"
--     ON user_predictions FOR SELECT
--     USING (auth.uid() = user_id);

-- =======================
-- VISTAS ÚTILES
-- =======================

-- Vista: Predicciones con información completa
CREATE OR REPLACE VIEW predictions_full AS
SELECT 
    p.id,
    p.compound_id,
    c.compound_id as original_id,
    p.smiles,
    c.tm_real,
    p.tm_pred,
    ABS(c.tm_real - p.tm_pred) as absolute_error,
    c.dataset_type,
    p.model_version,
    p.created_at
FROM predictions p
LEFT JOIN compounds c ON p.compound_id = c.id;

-- Vista: Estadísticas del modelo
CREATE OR REPLACE VIEW model_statistics AS
SELECT 
    COUNT(*) as total_predictions,
    AVG(ABS(tm_real - tm_pred)) as mae,
    SQRT(AVG(POWER(tm_real - tm_pred, 2))) as rmse,
    MIN(tm_pred) as min_prediction,
    MAX(tm_pred) as max_prediction,
    AVG(tm_pred) as avg_prediction
FROM predictions_full
WHERE tm_real IS NOT NULL;

-- Vista: Distribución por rangos de temperatura
CREATE OR REPLACE VIEW temperature_distribution AS
SELECT 
    CASE 
        WHEN tm_pred < 200 THEN '<200K'
        WHEN tm_pred BETWEEN 200 AND 250 THEN '200-250K'
        WHEN tm_pred BETWEEN 250 AND 300 THEN '250-300K'
        WHEN tm_pred BETWEEN 300 AND 350 THEN '300-350K'
        WHEN tm_pred BETWEEN 350 AND 400 THEN '350-400K'
        WHEN tm_pred BETWEEN 400 AND 450 THEN '400-450K'
        ELSE '>450K'
    END as temperature_range,
    COUNT(*) as count,
    ROUND((COUNT(*) * 100.0 / (SELECT COUNT(*) FROM predictions)), 2) as percentage
FROM predictions
GROUP BY temperature_range
ORDER BY 
    CASE temperature_range
        WHEN '<200K' THEN 1
        WHEN '200-250K' THEN 2
        WHEN '250-300K' THEN 3
        WHEN '300-350K' THEN 4
        WHEN '350-400K' THEN 5
        WHEN '400-450K' THEN 6
        ELSE 7
    END;

-- =======================
-- COMENTARIOS
-- =======================
COMMENT ON TABLE compounds IS 'Dataset completo de compuestos con SMILES y temperaturas de fusión reales';
COMMENT ON TABLE predictions IS 'Predicciones del modelo ChemProp para cada compuesto';
COMMENT ON TABLE user_predictions IS 'Predicciones personalizadas realizadas por usuarios';
COMMENT ON TABLE model_metadata IS 'Metadatos y métricas de los modelos ML';
COMMENT ON TABLE statistics_cache IS 'Caché de estadísticas pre-calculadas para mejorar performance';
