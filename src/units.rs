use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub struct Quantity {
    pub value: f64,
    pub unit: String,
}

impl Quantity {
    pub fn new(value: f64, unit: &str) -> Self {
        Self { value, unit: unit.to_string() }
    }

    pub fn convert_to(&self, target_unit: &str) -> Option<Quantity> {
        if self.unit == target_unit {
            return Some(self.clone());
        }
        
        // Basic conversion registry
        let factor = match (self.unit.as_str(), target_unit) {
            ("Meter", "Kilometer") => 0.001,
            ("Kilometer", "Meter") => 1000.0,
            ("Second", "Minute") => 1.0 / 60.0,
            ("Minute", "Second") => 60.0,
            ("Celsius", "Kelvin") => return Some(Quantity::new(self.value + 273.15, target_unit)),
            ("Kelvin", "Celsius") => return Some(Quantity::new(self.value - 273.15, target_unit)),
            _ => return None,
        };
        
        Some(Quantity::new(self.value * factor, target_unit))
    }
}

impl fmt::Display for Quantity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.value, self.unit)
    }
}

// Math operations with unit checking
impl std::ops::Add for Quantity {
    type Output = Result<Quantity, String>;
    fn add(self, rhs: Self) -> Self::Output {
        if self.unit == rhs.unit {
            Ok(Quantity::new(self.value + rhs.value, &self.unit))
        } else {
            // Attempt conversion
            if let Some(converted) = rhs.convert_to(&self.unit) {
                Ok(Quantity::new(self.value + converted.value, &self.unit))
            } else {
                Err(format!("Incompatible units: {} and {}", self.unit, rhs.unit))
            }
        }
    }
}

impl std::ops::Mul<f64> for Quantity {
    type Output = Quantity;
    fn mul(self, rhs: f64) -> Self::Output {
        Quantity::new(self.value * rhs, &self.unit)
    }
}
