#include <fstream>
#include <string>
#include <algorithm>
#include <vector>
#include "hmm.h"

using namespace std;

#define MAX_MODEL 10

void test(HMM *models, int n_model, vector<string> seqs, string out_path);
double viterbi(HMM model, string seq);
void load_seqs(string seq_path, vector<string> &seqs);


int main(int argc, char *argv[])
{
	string list_path = argv[1];
	string test_path = argv[2];
	string out_path = argv[3];
	int n_model;
	HMM models[MAX_MODEL];
	vector<string> seqs;
	
	n_model = load_models(list_path.c_str(), models, MAX_MODEL);
	load_seqs(test_path, seqs);

	test(models, n_model, seqs, out_path);

	return 0;
}

void test(HMM *models, int n_model, vector<string> seqs, string out_path)
{
	fstream fd(out_path, fstream::out);

	for (auto it = seqs.begin(); it != seqs.end(); it++) {
		string seq = *it;
		int max_id;
		double max_p = 0;
		
		for (int i = 0; i < n_model; i++) {
			double ret = viterbi(models[i], seq);
			if (ret > max_p) {
				max_p = ret;
				max_id = i;
			}
		}
		fd << models[max_id].model_name << ' ' << max_p << endl;
	}
}

double viterbi(HMM model, string seq)
{
	double delta[2][MAX_STATE];
	double *term;
	
	for (int i = 0; i < model.state_num; i++)
		delta[0][i] = model.initial[i] * model.observation[seq[0] - 'A'][i];

	for (int t = 1; t < seq.length(); t++) {
		for (int j = 0; j < model.state_num; j++) {
			double tmp = 0;
			for (int i = 0; i < model.state_num; i++)
				tmp = max(tmp, delta[(t + 1) % 2][i] * model.transition[i][j]);
			delta[t % 2][j] = tmp * model.observation[seq[t] - 'A'][j];
		}
		term = delta[t % 2];
	}

	return *max_element(term, term + model.state_num);	
}

void load_seqs(string seq_path, vector<string> &seqs)
{
	fstream fs(seq_path, fstream::in);
	string line;

	while (fs >> line) {
		seqs.push_back(line);
	}
}
