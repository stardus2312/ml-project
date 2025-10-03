// Drug Quality Classification Web Application
class DrugClassificationApp {
    constructor() {
        this.initializeElements();
        this.bindEvents();
        this.initializeApp();
    }

    initializeElements() {
        // Model settings
        this.architectureSelect = document.getElementById('architecture');
        this.deviceSelect = document.getElementById('device');
        
        // Input elements
        this.genericNameInput = document.getElementById('genericName');
        this.predictBtn = document.getElementById('predictBtn');
        
        // Output elements
        this.outputGenericName = document.getElementById('outputGenericName');
        this.outputMedicine = document.getElementById('outputMedicine');
        this.outputCategory = document.getElementById('outputCategory');
        this.outputAccuracy = document.getElementById('outputAccuracy');
    }

    bindEvents() {
        this.predictBtn.addEventListener('click', () => this.handlePrediction());
        this.genericNameInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.handlePrediction();
            }
        });
        
        // Model settings change events
        this.architectureSelect.addEventListener('change', () => this.updateModelSettings());
        this.deviceSelect.addEventListener('change', () => this.updateModelSettings());
    }

    initializeApp() {
        // Initialize with default values
        this.updateModelSettings();
        this.clearOutputs();
        
        // Add some sample data for demonstration
        this.sampleDrugs = [
            {
                genericName: "Acetaminophen",
                medicine: "Tylenol",
                category: "Analgesic",
                accuracy: "94.2%"
            },
            {
                genericName: "Ibuprofen",
                medicine: "Advil",
                category: "NSAID",
                accuracy: "91.8%"
            },
            {
                genericName: "Aspirin",
                medicine: "Bayer",
                category: "Antiplatelet",
                accuracy: "96.5%"
            },
            {
                genericName: "Metformin",
                medicine: "Glucophage",
                category: "Antidiabetic",
                accuracy: "89.3%"
            },
            {
                genericName: "Lisinopril",
                medicine: "Prinivil",
                category: "ACE Inhibitor",
                accuracy: "92.7%"
            }
        ];
    }

    updateModelSettings() {
        const architecture = this.architectureSelect.value;
        const device = this.deviceSelect.value;
        
        console.log(`Model Settings Updated: Architecture=${architecture}, Device=${device}`);
        
        // You can add logic here to actually update the ML model settings
        // For now, we'll just log the changes
    }

    async handlePrediction() {
        const genericName = this.genericNameInput.value.trim();
        
        if (!genericName) {
            this.showError('Please enter a drug generic name');
            return;
        }

        // Show loading state
        this.setLoadingState(true);
        
        try {
            // Simulate API call delay
            await this.delay(2000);
            
            // Get prediction result
            const result = this.getPredictionResult(genericName);
            
            // Display results
            this.displayResults(result);
            
            // Show success animation
            this.showSuccessAnimation();
            
        } catch (error) {
            this.showError('Prediction failed. Please try again.');
            console.error('Prediction error:', error);
        } finally {
            this.setLoadingState(false);
        }
    }

    getPredictionResult(genericName) {
        // Simulate ML prediction logic
        // In a real application, this would call your ML model API
        
        const normalizedName = genericName.toLowerCase();
        
        // Check if it's a known drug
        const knownDrug = this.sampleDrugs.find(drug => 
            drug.genericName.toLowerCase() === normalizedName
        );
        
        if (knownDrug) {
            return knownDrug;
        }
        
        // Generate a simulated result for unknown drugs
        return this.generateSimulatedResult(genericName);
    }

    generateSimulatedResult(genericName) {
        const categories = [
            "Analgesic", "Antibiotic", "Antihistamine", "Antidepressant",
            "Antihypertensive", "Antidiabetic", "NSAID", "Anticoagulant",
            "Bronchodilator", "Diuretic", "Antacid", "Antifungal"
        ];
        
        const medicines = [
            "Generic Brand", "PharmaCorp", "MediLife", "HealthPlus",
            "BioMed", "CureMax", "VitaCare", "Wellness Pro"
        ];
        
        const randomCategory = categories[Math.floor(Math.random() * categories.length)];
        const randomMedicine = medicines[Math.floor(Math.random() * medicines.length)];
        const randomAccuracy = (85 + Math.random() * 10).toFixed(1) + '%';
        
        return {
            genericName: genericName,
            medicine: randomMedicine,
            category: randomCategory,
            accuracy: randomAccuracy
        };
    }

    displayResults(result) {
        // Animate the results display
        this.animateOutput(this.outputGenericName, result.genericName);
        
        setTimeout(() => {
            this.animateOutput(this.outputMedicine, result.medicine);
        }, 200);
        
        setTimeout(() => {
            this.animateOutput(this.outputCategory, result.category);
        }, 400);
        
        setTimeout(() => {
            this.animateOutput(this.outputAccuracy, result.accuracy);
        }, 600);
    }

    animateOutput(element, value) {
        element.style.opacity = '0';
        element.style.transform = 'translateY(10px)';
        
        setTimeout(() => {
            element.textContent = value;
            element.style.opacity = '1';
            element.style.transform = 'translateY(0)';
            element.classList.add('success');
            
            setTimeout(() => {
                element.classList.remove('success');
            }, 600);
        }, 100);
    }

    setLoadingState(isLoading) {
        if (isLoading) {
            this.predictBtn.disabled = true;
            this.predictBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
            this.predictBtn.classList.add('loading');
            this.clearOutputs();
        } else {
            this.predictBtn.disabled = false;
            this.predictBtn.innerHTML = '<i class="fas fa-pills"></i> Prediction';
            this.predictBtn.classList.remove('loading');
        }
    }

    clearOutputs() {
        const outputs = [this.outputGenericName, this.outputMedicine, this.outputCategory, this.outputAccuracy];
        outputs.forEach(output => {
            output.textContent = '-';
            output.style.opacity = '0.5';
        });
    }

    showSuccessAnimation() {
        // Add success animation to the prediction section
        const predictionSection = document.querySelector('.prediction-section');
        predictionSection.classList.add('success');
        
        setTimeout(() => {
            predictionSection.classList.remove('success');
        }, 1000);
    }

    showError(message) {
        // Create and show error message
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.textContent = message;
        errorDiv.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: #ff4757;
            color: white;
            padding: 15px 20px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(255, 71, 87, 0.3);
            z-index: 1000;
            animation: slideIn 0.3s ease-out;
        `;
        
        document.body.appendChild(errorDiv);
        
        setTimeout(() => {
            errorDiv.style.animation = 'slideOut 0.3s ease-in';
            setTimeout(() => {
                document.body.removeChild(errorDiv);
            }, 300);
        }, 3000);
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Add CSS animations for error messages
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
    
    .error-message {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 500;
    }
`;
document.head.appendChild(style);

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new DrugClassificationApp();
    
    // Add some interactive features
    addInteractiveFeatures();
});

function addInteractiveFeatures() {
    // Add hover effects to prediction fields
    const predictionFields = document.querySelectorAll('.field-output');
    predictionFields.forEach(field => {
        field.addEventListener('mouseenter', () => {
            if (field.textContent !== '-') {
                field.style.transform = 'translateY(-2px)';
                field.style.boxShadow = '0 4px 12px rgba(102, 126, 234, 0.2)';
            }
        });
        
        field.addEventListener('mouseleave', () => {
            field.style.transform = 'translateY(0)';
            field.style.boxShadow = 'none';
        });
    });
    
    // Add click to copy functionality
    predictionFields.forEach(field => {
        field.addEventListener('click', () => {
            if (field.textContent !== '-') {
                navigator.clipboard.writeText(field.textContent).then(() => {
                    showCopyNotification();
                });
            }
        });
    });
    
    // Add keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.key === 'Enter') {
            document.getElementById('predictBtn').click();
        }
    });
}

function showCopyNotification() {
    const notification = document.createElement('div');
    notification.textContent = 'Copied to clipboard!';
    notification.style.cssText = `
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: #2ed573;
        color: white;
        padding: 10px 20px;
        border-radius: 6px;
        font-size: 14px;
        z-index: 1000;
        animation: fadeInOut 2s ease-in-out;
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        document.body.removeChild(notification);
    }, 2000);
}

// Add fade animation for copy notification
const copyStyle = document.createElement('style');
copyStyle.textContent = `
    @keyframes fadeInOut {
        0%, 100% { opacity: 0; transform: translate(-50%, -50%) scale(0.8); }
        20%, 80% { opacity: 1; transform: translate(-50%, -50%) scale(1); }
    }
`;
document.head.appendChild(copyStyle);

