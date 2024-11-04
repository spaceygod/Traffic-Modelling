from sklearn.mixture import GaussianMixture
import numpy as np

def fit_gmm_to_travel_times(all_car_reach_times):
    for delta, car_reach_times in all_car_reach_times:
        print(f"\nCapacity Multiplier: {delta}")
        reach_times = np.array(car_reach_times).reshape(-1, 1)  # Reshape for sklearn
        lowest_bic = np.infty
        best_n_components = None
        bic_list = []
        for n_components in range(2, 7):
            gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
            gmm.fit(reach_times)
            bic = gmm.bic(reach_times)
            bic_list.append(bic)
            print(f"\nNumber of Components: {n_components}")
            print(f"BIC: {bic:.2f}")
            for i in range(n_components):
                weight = gmm.weights_[i]
                mean = gmm.means_[i][0]
                variance = gmm.covariances_[i][0][0]
                print(f"Component {i+1}: Weight={weight:.4f}, Mean={mean:.4f}, Variance={variance:.4f}")
            if bic < lowest_bic:
                lowest_bic = bic
                best_n_components = n_components
        print(f"\nBest model for capacity multiplier {delta} has {best_n_components} components with BIC: {lowest_bic:.2f}")

# Same function but prints the terms of the BIC seprately for better understanding
def fit_gmm_with_bic_terms(all_car_reach_times, max_components=6):
    for delta, car_reach_times in all_car_reach_times:
        print(f"\nCapacity Multiplier: {delta}")
        reach_times = np.array(car_reach_times).reshape(-1, 1)  # Reshape for sklearn
        
        # Track the lowest BIC and the best number of components
        lowest_bic = np.infty
        best_n_components = None
        
        for n_components in range(2, max_components + 1):
            gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
            gmm.fit(reach_times)
            
            # Calculate BIC and separate its terms
            log_likelihood = gmm.score(reach_times) * len(reach_times)  # Total log-likelihood
            k = n_components * (1 + reach_times.shape[1] + reach_times.shape[1] * (reach_times.shape[1] + 1) / 2) - 1  # Parameters
            penalty_term = k * np.log(len(reach_times))
            bic = -2 * log_likelihood + penalty_term
            
            print(f"\nNumber of Components: {n_components}")
            print(f"Log-Likelihood Term (-2 * log-likelihood): {-2 * log_likelihood:.2f}")
            print(f"Penalty Term (k * ln(N)): {penalty_term:.2f}")
            print(f"Total BIC: {bic:.2f}")
            
            # Print GMM parameters for each component
            for i in range(n_components):
                weight = gmm.weights_[i]
                mean = gmm.means_[i][0]
                variance = gmm.covariances_[i][0][0]
                print(f"Component {i+1}: Weight={weight:.4f}, Mean={mean:.4f}, Variance={variance:.4f}")
            
            # Update the best model if the BIC is the lowest
            if bic < lowest_bic:
                lowest_bic = bic
                best_n_components = n_components

        print(f"\nBest model for capacity multiplier {delta} has {best_n_components} components with BIC: {lowest_bic:.2f}")

