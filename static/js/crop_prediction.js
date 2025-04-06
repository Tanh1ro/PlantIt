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
    const errors = [];
    
    for (let key in parameterRanges) {
        if (!(key in params) || params[key] === '') {
            errors.push({
                field: key,
                message: `Please enter the ${key} value`
            });
            continue;
        }
        
        let value = parseFloat(params[key]);
        if (isNaN(value)) {
            errors.push({
                field: key,
                message: `Please enter a valid number for ${key}`
            });
            continue;
        }
        
        const range = parameterRanges[key];
        if (value < range.min || value > range.max) {
            errors.push({
                field: key,
                message: `${key} should be between ${range.min} and ${range.max} ${range.unit}`
            });
        }
    }
    
    return {
        isValid: errors.length === 0,
        errors: errors
    };
}

// Function to display validation message
function showValidationMessage(message, isError = true, field = null) {
    const messageDiv = document.getElementById('validation-message');
    messageDiv.textContent = message;
    messageDiv.className = `alert alert-${isError ? 'danger' : 'success'} show`;
    
    if (field) {
        const input = document.querySelector(`[name="${field}"]`);
        if (input) {
            input.classList.add('is-invalid');
            const validationMessage = input.closest('.form-group').querySelector('.validation-message');
            validationMessage.textContent = message;
            validationMessage.classList.add('show');
        }
    }
}

// Function to handle form submission
async function handleFormSubmit(event) {
    event.preventDefault();
    
    const formData = new FormData(event.target);
    const params = {};
    
    for (let key in parameterRanges) {
        params[key] = formData.get(key);
    }
    
    const validation = validateCropParameters(params);
    
    if (validation.isValid) {
        showValidationMessage('Processing your request...', false);
        
        try {
            const response = await fetch('/crop-prediction', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(params)
            });

            if (!response.ok) {
                throw new Error('Failed to get crop recommendations');
            }

            const result = await response.json();
            
            // Show results container
            const resultsContainer = document.getElementById('resultsContainer');
            resultsContainer.style.display = 'block';

            // Update predicted crop name
            document.getElementById('predicted-crop').textContent = result.predicted_crop;

            // Update optimal growing conditions
            document.getElementById('temp-range').textContent = result.temperature_range || 'Not specified';
            document.getElementById('ph-range').textContent = result.ph_range || 'Not specified';
            document.getElementById('rainfall-req').textContent = result.rainfall_requirements || 'Not specified';
            document.getElementById('soil-type').textContent = result.soil_type || 'Not specified';

            // Update growing tips
            document.getElementById('planting-season').textContent = result.planting_season || 'Not specified';
            document.getElementById('harvest-time').textContent = result.harvest_time || 'Not specified';
            document.getElementById('water-req').textContent = result.water_requirements || 'Not specified';
            document.getElementById('fertilizer-needs').textContent = result.fertilizer_needs || 'Not specified';

            // Update alternative crops table
            const tbody = document.getElementById('alternative-crops');
            tbody.innerHTML = '';

            if (result.alternative_crops && result.alternative_crops.length > 0) {
                result.alternative_crops.forEach(crop => {
                    const row = document.createElement('tr');
                    row.innerHTML = `
                        <td>${crop.name}</td>
                        <td>${crop.score}%</td>
                        <td>${crop.temperature_range}</td>
                        <td>${crop.ph_range}</td>
                        <td>${crop.rainfall_requirements}</td>
                    `;
                    tbody.appendChild(row);
                });
            } else {
                tbody.innerHTML = '<tr><td colspan="5" class="text-center">No alternative crops available</td></tr>';
            }

            // Scroll to results
            resultsContainer.scrollIntoView({ behavior: 'smooth' });
        } catch (error) {
            showValidationMessage(error.message, true);
        }
    } else {
        // Clear previous validation messages
        document.querySelectorAll('.validation-message').forEach(el => {
            el.textContent = '';
            el.classList.remove('show');
        });
        document.querySelectorAll('.is-invalid').forEach(el => {
            el.classList.remove('is-invalid');
        });
        
        // Show new validation messages
        validation.errors.forEach(error => {
            showValidationMessage(error.message, true, error.field);
        });
    }
}

// Function to validate individual input
function validateInput(input) {
    const value = parseFloat(input.value);
    const key = input.name;
    const range = parameterRanges[key];
    const validationMessage = input.closest('.form-group').querySelector('.validation-message');
    
    if (input.value === '') {
        input.classList.add('is-invalid');
        validationMessage.textContent = 'This field is required';
        validationMessage.classList.add('show');
        return false;
    }
    
    if (isNaN(value)) {
        input.classList.add('is-invalid');
        validationMessage.textContent = 'Please enter a valid number';
        validationMessage.classList.add('show');
        return false;
    }

    if (value < range.min || value > range.max) {
        input.classList.add('is-invalid');
        validationMessage.textContent = `Value must be between ${range.min} and ${range.max} ${range.unit}`;
        validationMessage.classList.add('show');
        return false;
    }

    input.classList.remove('is-invalid');
    validationMessage.textContent = '';
    validationMessage.classList.remove('show');
    return true;
}

// Add event listeners when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('cropForm');
    if (!form) return;

    const submitBtn = document.getElementById('submitBtn');
    const validationMessage = document.getElementById('validation-message');
    const resultsContainer = document.getElementById('resultsContainer');

    // Parameter ranges
    const ranges = {
        nitrogen: { min: 10, max: 300 },
        phosphorus: { min: 5, max: 150 },
        potassium: { min: 50, max: 400 },
        temperature: { min: 5, max: 40 },
        humidity: { min: 20, max: 100 },
        soilPH: { min: 4.5, max: 8.5 },
        rainfall: { min: 10, max: 5000 }
    };

    // Validate a single parameter
    function validateParameter(value, paramName) {
        const range = ranges[paramName];
        if (!value) {
            return `${paramName.charAt(0).toUpperCase() + paramName.slice(1)} is required`;
        }
        if (isNaN(value)) {
            return `${paramName.charAt(0).toUpperCase() + paramName.slice(1)} must be a number`;
        }
        if (value < range.min || value > range.max) {
            return `${paramName.charAt(0).toUpperCase() + paramName.slice(1)} must be between ${range.min} and ${range.max}`;
        }
        return null;
    }

    // Validate all parameters
    function validateCropParameters() {
        const errors = [];
        const formData = new FormData(form);
        
        for (const [param, value] of formData.entries()) {
            const error = validateParameter(value, param);
            if (error) {
                errors.push(error);
                const input = document.getElementById(param);
                if (input) {
                    input.classList.add('is-invalid');
                    input.classList.remove('is-valid');
                }
            } else {
                const input = document.getElementById(param);
                if (input) {
                    input.classList.add('is-valid');
                    input.classList.remove('is-invalid');
                }
            }
        }

        return {
            isValid: errors.length === 0,
            errors: errors
        };
    }

    // Show validation message
    function showValidationMessage(message, type = 'error') {
        if (!validationMessage) return;
        validationMessage.textContent = message;
        validationMessage.className = `alert alert-${type === 'error' ? 'danger' : 'success'} d-block`;
        validationMessage.style.display = 'block';
    }

    // Clear validation message
    function clearValidationMessage() {
        if (!validationMessage) return;
        validationMessage.textContent = '';
        validationMessage.className = 'alert d-none';
        validationMessage.style.display = 'none';
    }

    // Handle form submission
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        clearValidationMessage();

        const validation = validateCropParameters();
        
        if (!validation.isValid) {
            showValidationMessage(validation.errors.join('\n'));
            return;
        }

        if (submitBtn) {
            submitBtn.disabled = true;
            submitBtn.textContent = 'Processing...';
        }

        try {
            const formData = new FormData(form);
            const response = await fetch(form.action, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();
            
            if (data.error) {
                showValidationMessage(data.error);
            } else {
                showValidationMessage('Crop prediction successful!', 'success');
                if (resultsContainer) {
                    resultsContainer.style.display = 'block';
                }
            }
        } catch (error) {
            showValidationMessage('An error occurred while processing your request. Please try again.');
            console.error('Error:', error);
        } finally {
            if (submitBtn) {
                submitBtn.disabled = false;
                submitBtn.textContent = 'Get Crop Recommendations';
            }
        }
    });

    // Add input event listeners for real-time validation
    const inputs = form.querySelectorAll('input[type="number"]');
    inputs.forEach(input => {
        input.addEventListener('input', function() {
            const paramName = this.id;
            const value = this.value;
            const error = validateParameter(value, paramName);
            
            if (error) {
                this.classList.add('is-invalid');
                this.classList.remove('is-valid');
            } else {
                this.classList.add('is-valid');
                this.classList.remove('is-invalid');
            }
        });
    });
}); 