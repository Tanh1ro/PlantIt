document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('cropForm');
    const validationMessage = document.getElementById('validation-message');
    const submitBtn = document.getElementById('submitBtn');
    const resultsContainer = document.getElementById('resultsContainer');

    // Add input validation
    const inputs = form.querySelectorAll('input[type="number"]');
    inputs.forEach(input => {
        // Add input validation
        input.addEventListener('input', function() {
            validateInput(this);
        });

        // Add blur validation
        input.addEventListener('blur', function() {
            validateInput(this);
        });
    });

    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Validate all inputs
        let isValid = true;
        inputs.forEach(input => {
            if (!validateInput(input)) {
                isValid = false;
            }
        });

        if (!isValid) {
            showMessage('Please correct the errors in the form.', 'danger');
            return;
        }

        // Disable submit button and show loading state
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...';
        
        try {
            // Get form data
            const formData = new FormData(form);
            const data = {};
            formData.forEach((value, key) => {
                data[key] = parseFloat(value);
            });

            // Send data to backend
            const response = await fetch('/crop-prediction', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            });

            if (!response.ok) {
                throw new Error('Failed to get crop recommendations');
            }

            const result = await response.json();

            // Show results container
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

            // Show success message
            showMessage('Crop recommendations generated successfully!', 'success');

            // Scroll to results
            resultsContainer.scrollIntoView({ behavior: 'smooth' });
        } catch (error) {
            showMessage(error.message, 'danger');
        } finally {
            // Re-enable submit button
            submitBtn.disabled = false;
            submitBtn.textContent = 'Get Crop Recommendations';
        }
    });

    function validateInput(input) {
        const value = parseFloat(input.value);
        const min = parseFloat(input.min);
        const max = parseFloat(input.max);
        const unit = input.getAttribute('data-unit');
        const validationMessage = input.closest('.form-group').querySelector('.validation-message');
        
        if (input.value === '') {
            validationMessage.textContent = 'This field is required';
            validationMessage.classList.add('show');
            input.classList.add('is-invalid');
            return false;
        }
        
        if (isNaN(value)) {
            validationMessage.textContent = 'Please enter a valid number';
            validationMessage.classList.add('show');
            input.classList.add('is-invalid');
            return false;
        }

        if (value < min || value > max) {
            validationMessage.textContent = `Value must be between ${min} and ${max} ${unit}`;
            validationMessage.classList.add('show');
            input.classList.add('is-invalid');
            return false;
        }

        validationMessage.textContent = '';
        validationMessage.classList.remove('show');
        input.classList.remove('is-invalid');
        return true;
    }

    function showMessage(message, type) {
        validationMessage.className = `alert alert-${type}`;
        validationMessage.textContent = message;
        validationMessage.classList.remove('d-none');
    }
}); 