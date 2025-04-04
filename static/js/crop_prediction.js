// Ranges for crop parameters
const parameterRanges = {
    nitrogen: { min: 10, max: 300, unit: 'mg/kg' },
    phosphorus: { min: 5, max: 150, unit: 'mg/kg' },
    potassium: { min: 50, max: 400, unit: 'mg/kg' },
    temperature: { min: 5, max: 40, unit: '°C' },
    humidity: { min: 20, max: 100, unit: '%' },
    soilPH: { min: 4.5, max: 8.5, unit: 'pH' },
    rainfall: { min: 10, max: 5000, unit: 'mm/year' }
};

// Function to validate crop parameters
function validateCropParameters(params) {
    for (let key in parameterRanges) {
        if (!(key in params)) {
            return {
                isValid: false,
                message: `Please enter the ${key}`
            };
        }
        
        let value = parseFloat(params[key]);
        if (isNaN(value)) {
            return {
                isValid: false,
                message: `Please enter a valid number for ${key}`
            };
        }
        
        const range = parameterRanges[key];
        if (value < range.min || value > range.max) {
            return {
                isValid: false,
                message: `${key} should be between ${range.min} and ${range.max} ${range.unit}`
            };
        }
    }
    return {
        isValid: true,
        message: "All parameters are valid"
    };
}

// Function to display validation message
function showValidationMessage(message, isError = true) {
    const messageDiv = document.getElementById('validation-message');
    if (!messageDiv) {
        const div = document.createElement('div');
        div.id = 'validation-message';
        document.querySelector('form').insertBefore(div, document.querySelector('button[type="submit"]'));
    }
    
    const messageElement = document.getElementById('validation-message');
    messageElement.textContent = message;
    messageElement.className = isError ? 'error-message' : 'success-message';
}

// Function to handle form submission
function handleFormSubmit(event) {
    event.preventDefault();
    
    const formData = new FormData(event.target);
    const params = {};
    
    for (let key in parameterRanges) {
        params[key] = formData.get(key);
    }
    
    const validation = validateCropParameters(params);
    
    if (validation.isValid) {
        showValidationMessage(validation.message, false);
        // Submit the form if validation passes
        event.target.submit();
    } else {
        showValidationMessage(validation.message, true);
    }
}

// Add event listeners when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    const form = document.querySelector('form');
    if (form) {
        form.addEventListener('submit', handleFormSubmit);
    }
    
    // Add input event listeners for real-time validation
    for (let key in parameterRanges) {
        const input = document.querySelector(`[name="${key}"]`);
        if (input) {
            input.addEventListener('input', function() {
                const value = parseFloat(this.value);
                const range = parameterRanges[key];
                
                if (isNaN(value)) {
                    this.setCustomValidity(`Please enter a valid number for ${key}`);
                } else if (value < range.min || value > range.max) {
                    this.setCustomValidity(`${key} should be between ${range.min} and ${range.max} ${range.unit}`);
                } else {
                    this.setCustomValidity('');
                }
                
                this.reportValidity();
            });
        }
    }
}); 