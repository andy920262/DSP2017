#include <fstream>
#include <vector>
#include <string>
#include <numeric>
#include <iostream>
#include "hmm.h"
using namespace std;

void train(HMM &model, int num_iters, vector<string> seqs);
void forward_procedure(HMM &model, double alpha[][MAX_SEQ], string seq);
void backward_procedure(HMM &model, double beta[][MAX_SEQ], string seq);
void calc_gamma(double gamma[][MAX_SEQ], double alpha[][MAX_SEQ], double beta[][MAX_SEQ], int n_state, int seql);
void calc_epsilon(double epsilon[][MAX_STATE][MAX_SEQ], HMM model, double alpha[][MAX_SEQ], double beta[][MAX_SEQ], string seq, int n_state, int seql);
void load_seqs(string seq_path, vector<string> &seqs);

int main(int argc, char *argv[])
{
	int num_iters = stoi(string(argv[1]));
	string init_path = argv[2];
	string seq_path = argv[3];
	string out_path = argv[4];
	HMM model;
	vector<string> seqs;

	loadHMM(&model, init_path.c_str());
	load_seqs(seq_path, seqs);

	train(model, num_iters, seqs);

	dumpHMM(open_or_die(out_path.c_str(), "w"), &model);

	return 0;
}

/**********************/
/* Training Functions */
/**********************/

void train(HMM &model, int num_iters, vector<string> seqs)
{
	int seql = seqs[0].length();
	int n_state = model.state_num;

	for (int iter = 1; iter <= num_iters; iter++) {
		double alpha[MAX_STATE][MAX_SEQ] = {};
		double beta[MAX_STATE][MAX_SEQ] = {};
		double gamma[MAX_STATE][MAX_SEQ] = {};
		double epsilon[MAX_STATE][MAX_STATE][MAX_SEQ] = {};
	
		double p_sum[MAX_STATE] = {};
		double a_n[MAX_STATE][MAX_STATE] = {};
		double a_d[MAX_STATE] = {};
		double b_n[MAX_STATE][MAX_STATE] = {};
		double b_d[MAX_STATE] = {};
		
		for (auto it = seqs.begin(); it != seqs.end(); it++) {
			string seq = *it;
			
			forward_procedure(model, alpha, seq);
			backward_procedure(model, beta, seq);
			calc_gamma(gamma, alpha, beta, n_state, seql);
			calc_epsilon(epsilon, model, alpha, beta, seq, n_state, seql);
			
			for (int i = 0; i < n_state; i++) {
				p_sum[i] += gamma[i][0];
				for (int t = 0; t < (seql - 1); t++) {
					a_d[i] += gamma[i][t];
					for (int j = 0; j < n_state; j++)
						a_n[i][j] += epsilon[i][j][t];
				}
				for (int t = 0; t < seql; t++) {
					b_d[i] += gamma[i][t];
					b_n[seq[t] - 'A'][i] += gamma[i][t];
				}
			}
		}

		/* Update model */
		for (int i = 0; i < model.state_num; i++) {
			model.initial[i] = p_sum[i] / seqs.size();
		}
		for (int i = 0; i < model.state_num; i++) {
			for (int j = 0; j < model.state_num; j++) {
				model.transition[i][j] = a_n[i][j] / a_d[i];
				model.observation[j][i] = b_n[j][i] / b_d[i];
			}
		}
	}
}

void forward_procedure(HMM &model, double alpha[][MAX_SEQ], string seq)
{
	int seql = seq.length();
	int n_state = model.state_num;

	for (int i = 0; i < n_state; i++)
		alpha[i][0] = model.initial[i] * model.observation[seq[0] - 'A'][i];
	for (int t = 1; t < seql; t++) {
		for (int i = 0; i < n_state; i++) {
			double sum = 0;
			for (int j = 0; j < n_state; j++)
				sum += alpha[j][t - 1] * model.transition[j][i];
			alpha[i][t] = model.observation[seq[t] - 'A'][i] * sum;
		}
	}
}

void backward_procedure(HMM &model, double beta[][MAX_SEQ], string seq)
{
	int seql = seq.length();
	int n_state = model.state_num;
	
	for (int i = 0; i < n_state; i++)
		beta[i][seql - 1] = 1;
	for (int t = seql - 2; t >= 0; t--) {
		for (int i = 0; i < n_state; i++) {
			double sum = 0;
			for (int j = 0; j < n_state; j++)
				sum += beta[j][t + 1] * model.transition[i][j] * model.observation[seq[t + 1] - 'A'][j];
			beta[i][t] = sum;
		}
	}
}

void calc_gamma(double gamma[][MAX_SEQ], double alpha[][MAX_SEQ], double beta[][MAX_SEQ], int n_state, int seql)
{
	for (int t = 0; t < seql; t++) {
		double sum = 0;
		for (int i = 0; i < n_state; i++)
			sum += alpha[i][t] * beta[i][t];
		for (int i = 0; i < n_state; i++) {
			gamma[i][t] = alpha[i][t] * beta[i][t] / sum;
		}
	}
}

void calc_epsilon(double epsilon[][MAX_STATE][MAX_SEQ], HMM model, double alpha[][MAX_SEQ], double beta[][MAX_SEQ], string seq, int n_state, int seql)
{
	for (int t = 0; t < (seql - 1); t++) {	
		double sum = 0;
		for (int i = 0; i < n_state; i++) {
			for (int j = 0; j < n_state; j++) {
				sum += alpha[i][t] * model.transition[i][j] * beta[j][t + 1] * model.observation[seq[t + 1] - 'A'][j];
			}
		}
		
		for (int i = 0; i < n_state; i++) {
			for (int j = 0; j < n_state; j++) {
				epsilon[i][j][t] = alpha[i][t] * model.transition[i][j] * beta[j][t + 1] * model.observation[seq[t + 1] - 'A'][j] / sum;
			}
		}
	}
}

void load_seqs(string seq_path, vector<string> &seqs)
{
	fstream fs(seq_path, fstream::in);
	string line;

	while (fs >> line) {
		seqs.push_back(line);
	}
}
